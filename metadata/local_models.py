#!/usr/bin/env python3
# /// script
# dependencies = [
#   "huggingface_hub>=1,<2",
# ]
# ///
"""Generate and validate local-models.csv.

A local model is a conversion-only copy (GGUF, MLX, AWQ, ...) of a models.csv
checkpoint. Fine-tuned, pruned, merged, distilled and other modified
descendants are excluded. Downloads are credited to the original parent's
organization and region, never the converter.

    uv run metadata/local_models.py            # regenerate from the Hugging Face API
    python metadata/local_models.py --validate # check the committed CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "local-models.csv"
OVERRIDES_PATH = REPO_ROOT / "metadata" / "local_models_overrides.json"
REGIONS_PATH = REPO_ROOT / "metadata" / "organization_regions.csv"
MODELS_PATH = REPO_ROOT / "models.csv"
EXTRA_MODELS_PATH = REPO_ROOT / "extra_models.csv"

# Lifetime counts never decrease, so admission is a pure function of current
# data: once a conversion qualifies it stays qualified.
MIN_LIFETIME_DOWNLOADS = 100_000
COLUMNS = (
    "org",
    "model",
    "modelId",
    "parent_model_id",
    "parent_org",
    "parent_region",
    "local_formats",
    "created_at",
)
FORMATS = (
    "gguf",
    "mlx",
    "awq",
    "gptq",
    "exl2",
    "onnx",
    "openvino",
    "hqq",
    "exl3",
    "bitsandbytes",
    "aqlm",
    "quanto",
    "ggml",
    "compressed-tensors",
    "auto-round",
    "modelopt",
)
FORMAT_ALIASES = {
    **{fmt: fmt for fmt in FORMATS},
    "mlx-lm": "mlx",
    "mlx-vlm": "mlx",
    "autoawq": "awq",
    "auto-awq": "awq",
    "auto-gptq": "gptq",
    "autogptq": "gptq",
    "auto_gptq": "gptq",
    "gptqmodel": "gptq",
    "onnxruntime": "onnx",
    "onnxruntime-web": "onnx",
    "onnxruntime-genai": "onnx",
    "exllamav3": "exl3",
    "exllama-v3": "exl3",
    "bits-and-bytes": "bitsandbytes",
    "bnb": "bitsandbytes",
    "optimum-quanto": "quanto",
    "autoround": "auto-round",
    "auto_round": "auto-round",
    "model optimizer": "modelopt",
}
# Name or tag markers of a modified descendant rather than a plain conversion.
MODIFICATION_WORDS = (
    "sft",
    "dpo",
    "orpo",
    "grpo",
    "rlhf",
    "finetune",
    "finetuned",
    "tuned",
    "lora",
    "qlora",
    "adapter",
    "merge",
    "merged",
    "reap",
    "pruned",
    "pruning",
    "abliterated",
    "abliteration",
    "obliterated",
    "heretic",
    "uncensored",
    "distill",
    "distilled",
)
MODIFICATION_TAGS = frozenset(
    {"reap", "expert-pruning", "pruned", "pruning", "abliterated", "abliteration", "heretic", "uncensored"}
)
# Tokens a converter may put around the parent's name (Q4_K_M, 4bit, imatrix, ...).
CONVERSION_WORDS = frozenset(FORMAT_ALIASES) | frozenset(
    """ggml llamacpp llama cpp quant quantized quantization quants quantisation
    converted conversion optimized optimised compatible weights safetensors fp bf f
    int q k m s l xl xxs xs iq i ud bpw bit bits g gs group size imatrix imat static
    dynamic hf cpu gpu cuda npu xpu ov web v model models exl exllama auto runtime lm
    vlm bnb nl w a x actorder true false desc act sym asym triton directml webgpu mxfp
    nvfp nf autoround round sharded qat""".split()
)


class CatalogError(ValueError):
    pass


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def read_model_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row["modelId"] for row in csv.DictReader(handle)]


def load_regions(path: Path = REGIONS_PATH) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {row["hf_org"].casefold(): row["region"] for row in csv.DictReader(handle)}


def load_overrides(path: Path = OVERRIDES_PATH) -> dict:
    overrides = json.loads(path.read_text())
    if set(overrides) != {"aliases", "include", "exclude"}:
        raise CatalogError("Overrides need exactly aliases, include and exclude")
    return overrides


def detect_formats(model: dict) -> list[str]:
    """Formats evidenced by library, tags, quantization config or file names."""
    found = set()
    for value in [model.get("library_name") or "", *model.get("tags", [])]:
        if value.casefold() in FORMAT_ALIASES:
            found.add(FORMAT_ALIASES[value.casefold()])
    quant = (model.get("config") or {}).get("quantization_config") or {}
    method = quant.get("quant_method") if isinstance(quant, dict) else None
    if isinstance(method, str) and method.casefold() in FORMAT_ALIASES:
        found.add(FORMAT_ALIASES[method.casefold()])
    for filename in model.get("siblings", []):
        lower = filename.casefold()
        if lower.endswith(".gguf"):
            found.add("gguf")
        elif lower.endswith(".onnx"):
            found.add("onnx")
        elif lower.endswith(".exl3"):
            found.add("exl3")
        elif re.search(r"(^|/)openvino[^/]*\.xml$", lower):
            found.add("openvino")
    return [fmt for fmt in FORMATS if fmt in found]


def base_models(model: dict, aliases: dict[str, str]) -> tuple[set[str], set[str]]:
    """Declared base checkpoints (alias-resolved) and base-model relations."""
    bases, relations = set(), set()
    for tag in model.get("tags", []):
        if tag.startswith("base_model:"):
            value = tag.removeprefix("base_model:")
            if value.count(":") == 1:
                relation, value = value.split(":")
                relations.add(relation.casefold())
            bases.add(value)
    card = model.get("card_data") or {}
    card_bases = card.get("base_model") or []
    bases.update([card_bases] if isinstance(card_bases, str) else card_bases)
    if isinstance(card.get("base_model_relation"), str):
        relations.add(card["base_model_relation"].casefold())
    folded = {key.casefold(): value for key, value in aliases.items()}
    return {folded.get(base.casefold(), base).casefold() for base in bases}, relations


def name_preserves_parent(model_id: str, parent_id: str) -> bool:
    """The parent's name survives, surrounded only by conversion modifiers."""
    org, name = model_id.split("/", 1)
    parent_org, parent_name = parent_id.split("/", 1)
    compact = _compact(name)
    # Converters often drop the Transformers "-hf" suffix and the Meta- prefix.
    candidates = [parent_name, re.sub(r"[._-]hf$", "", parent_name, flags=re.IGNORECASE)]
    if parent_org.casefold() == "meta-llama":
        candidates.append(re.sub(r"^meta[._-]", "", candidates[1], flags=re.IGNORECASE))
    parent = next((_compact(c) for c in candidates if _compact(c) and _compact(c) in compact), None)
    if parent is None:
        return False
    # Map the compact match back onto the original name to recover its modifiers.
    lowered = name.casefold()
    positions = [i for i, char in enumerate(lowered) if char.isascii() and char.isalnum()]
    start = compact.index(parent)
    before = lowered[: positions[start]]
    after = lowered[positions[start + len(parent) - 1] + 1 :]
    allowed = CONVERSION_WORDS | set(re.findall(r"[a-z]+", org.casefold() + " " + parent_org.casefold()))
    return all(token in allowed for token in re.findall(r"[a-z]+", before + " " + after))


def rejection(model: dict, parent_id: str, aliases: dict[str, str]) -> str | None:
    """Why a quantized child of parent_id is not a conversion-only local copy."""
    if not detect_formats(model):
        return "no_format_evidence"
    bases, relations = base_models(model, aliases)
    # Older first-party releases often declare no base model; their name and
    # publisher still identify the parent (checked below).
    official = model["id"].split("/")[0].casefold() == parent_id.split("/")[0].casefold()
    if bases != {parent_id.casefold()} and not (official and not bases):
        return "not_single_parent"
    if relations - {"quantized"}:
        return "modified_relation"
    name = model["id"].split("/", 1)[1].casefold()
    parent_name = parent_id.split("/", 1)[1].casefold()
    added = set(re.findall(r"[a-z]+", name)) - set(re.findall(r"[a-z]+", parent_name))
    if added & set(MODIFICATION_WORDS) or {t.casefold() for t in model.get("tags", [])} & MODIFICATION_TAGS:
        return "modified_name_or_tag"
    explicit = (model.get("card_data") or {}).get("base_model_relation")
    if explicit != "quantized" and not name_preserves_parent(model["id"], parent_id):
        return "name_not_corroborated"
    return None


def admitted(model: dict, parent_id: str) -> bool:
    official = model["id"].split("/")[0].casefold() == parent_id.split("/")[0].casefold()
    return official or (model.get("downloads_all_time") or 0) >= MIN_LIFETIME_DOWNLOADS


def _created(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date().isoformat()
    return date.isoformat(value)


def build_rows(
    candidates: Iterable[tuple[str, dict]],
    parents: list[str],
    excluded: set[str],
    regions: dict[str, str],
    overrides: dict,
) -> list[dict]:
    """candidates: (parent_id, model) pairs from the quantized-children query."""
    canonical = {pid.casefold(): pid for pid in parents}
    includes = {key.casefold(): value for key, value in overrides["include"].items()}
    excludes = {key.casefold() for key in overrides["exclude"]} | {e.casefold() for e in excluded}
    rows = {}
    for parent_id, model in candidates:
        key = model["id"].casefold()
        if key in excludes or key in rows:
            continue
        if key in includes:
            parent_id = canonical[includes[key]["parent"].casefold()]
            formats = includes[key].get("formats") or detect_formats(model)
        elif rejection(model, parent_id, overrides["aliases"]):
            continue
        else:
            formats = detect_formats(model)
        if not formats or not admitted(model, parent_id):
            continue
        org, name = model["id"].split("/", 1)
        parent_org = parent_id.split("/")[0]
        rows[key] = {
            "org": org,
            "model": name,
            "modelId": model["id"],
            "parent_model_id": parent_id,
            "parent_org": parent_org,
            "parent_region": regions[parent_org.casefold()],
            "local_formats": ";".join(formats),
            "created_at": _created(model.get("created_at")),
        }
    return sorted(rows.values(), key=lambda row: (row["modelId"].casefold(), row["modelId"]))


def write_catalog(rows: list[dict], path: Path = CATALOG_PATH) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_catalog(
    path: Path, parents: list[str], tracked: list[str], regions: dict[str, str]
) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise CatalogError(f"{path}: expected columns {','.join(COLUMNS)}")
        rows = list(reader)
    canonical = {pid.casefold() for pid in parents}
    tracked_keys = {mid.casefold() for mid in tracked}
    keys = [(row["modelId"].casefold(), row["modelId"]) for row in rows]
    if keys != sorted(keys) or len({k for k, _ in keys}) != len(keys):
        raise CatalogError(f"{path}: rows must be unique and sorted by modelId")
    for line, row in enumerate(rows, start=2):
        model_id, parent_id = row["modelId"], row["parent_model_id"]
        problems = [
            f"{row['org']}/{row['model']} != modelId" if f"{row['org']}/{row['model']}" != model_id else "",
            "already tracked" if model_id.casefold() in tracked_keys else "",
            "parent not in models.csv" if parent_id.casefold() not in canonical else "",
            "parent_org mismatch" if parent_id.split("/")[0] != row["parent_org"] else "",
            "parent_region mismatch"
            if regions.get(row["parent_org"].casefold()) != row["parent_region"]
            else "",
            "unknown or unordered formats"
            if not row["local_formats"]
            or [f for f in FORMATS if f in row["local_formats"].split(";")]
            != row["local_formats"].split(";")
            else "",
            "bad created_at" if row["created_at"] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["created_at"]) else "",
        ]
        if any(problems):
            raise CatalogError(f"{path}:{line}: {model_id}: {'; '.join(p for p in problems if p)}")
    return rows


def _to_dict(info) -> dict:
    return {
        "id": info.id,
        "tags": list(info.tags or []),
        "library_name": info.library_name,
        "downloads_all_time": getattr(info, "downloads_all_time", None),
        "created_at": info.created_at,
        "card_data": {
            key: info.card_data.get(key) for key in ("base_model", "base_model_relation")
        }
        if info.card_data
        else {},
        "config": info.config or {},
        "siblings": [s.rfilename for s in info.siblings or []],
    }


def fetch_candidates(parents: list[str], overrides: dict) -> list[tuple[str, dict]]:
    """Quantized children of each parent (and its renamed aliases), plus
    first-party repositories named as conversions of a parent."""
    import logging

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # malformed model-index noise
    api = HfApi()
    expand = ["tags", "library_name", "downloadsAllTime", "createdAt", "cardData"]
    names = {parent: [parent] for parent in parents}
    canonical = {pid.casefold(): pid for pid in parents}
    for alias, target in overrides["aliases"].items():
        if target.casefold() in canonical:
            names[canonical[target.casefold()]].append(alias)
    def listing(**query) -> list:
        for attempt in range(4):
            try:
                return list(api.list_models(expand=expand, **query))
            except HfHubHTTPError as exc:
                if exc.response is None or exc.response.status_code != 429 or attempt == 3:
                    raise
                print("  rate limited; waiting a minute", file=sys.stderr)
                time.sleep(60)

    candidates = []
    for index, parent in enumerate(parents, start=1):
        for name in names[parent]:
            listed = listing(filter=f"base_model:quantized:{name}")
            candidates.extend((parent, _to_dict(info)) for info in listed)
        if index % 100 == 0:
            print(f"  {index}/{len(parents)} parents, {len(candidates)} children", file=sys.stderr)
    # First-party conversions published before base_model tagging existed are
    # found by name among each parent organization's own repositories.
    by_org: dict[str, list[str]] = {}
    for parent in parents:
        by_org.setdefault(parent.split("/")[0], []).append(parent)
    for org, org_parents in sorted(by_org.items()):
        for info in listing(author=org):
            if info.id.casefold() in canonical:
                continue
            matches = [p for p in org_parents if name_preserves_parent(info.id, p)]
            if matches:
                candidates.append((max(matches, key=len), _to_dict(info)))
    # Reviewed includes may lack the quantized tag; fetch them directly.
    for model_id in overrides["include"]:
        candidates.append(("", _to_dict(api.model_info(model_id, expand=expand))))
    # Tags miss some formats (EXL3, ONNX, quant configs); look closer only at
    # repositories that would otherwise be admitted.
    for position, (parent, model) in enumerate(candidates):
        if parent and not detect_formats(model) and admitted(model, parent):
            info = api.model_info(model["id"], expand=[*expand, "config", "siblings"])
            candidates[position] = (parent, _to_dict(info))
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--validate", action="store_true", help="check the committed CSV only")
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    args = parser.parse_args(argv)
    parents = read_model_ids(MODELS_PATH)
    tracked = parents + read_model_ids(EXTRA_MODELS_PATH)
    regions = load_regions()
    try:
        if not args.validate:
            overrides = load_overrides()
            candidates = fetch_candidates(parents, overrides)
            write_catalog(build_rows(candidates, parents, set(tracked), regions, overrides), args.output)
        rows = validate_catalog(args.output, parents, tracked, regions)
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.output.name}: {len(rows)} local conversions of {len({r['parent_model_id'] for r in rows})} parents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
