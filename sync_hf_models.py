# /// script
# dependencies = [
#   "huggingface_hub>=1,<2",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent
CSV_NAMES = ("models.csv", "extra_models.csv")
SKIP_PIPELINE_TAGS = {
    "audio-classification",
    "audio-text-to-text",
    "audio-to-audio",
    "automatic-speech-recognition",
    "depth-estimation",
    "feature-extraction",
    "image-classification",
    "image-feature-extraction",
    "image-segmentation",
    "image-to-3d",
    "image-to-image",
    "image-to-video",
    "object-detection",
    "reinforcement-learning",
    "sentence-similarity",
    "text-classification",
    "text-ranking",
    "time-series-forecasting",
    "video-classification",
}
SKIP_KEYWORDS = {
    "guard",
    "guardian",
    "shield",
    "safeguard",
    "safety",
    "test",
    "dummy",
    "warmup",
    "classifier",
    "detector",
    "embedding",
    "embed",
    "forecast",
    "patchtst",
    "layout",
    "transcribe",
    "asr",
    "store",
}
ALLOW_UNTAGGED_TAGS = {"transformers", "llama.cpp", "vllm", "diffusers"}
HF_METADATA_SUFFIXES = (
    "config.json",
    "adapter_config.json",
    "model_index.json",
    "preprocessor_config.json",
    "processor_config.json",
)
WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".keras",
    ".msgpack",
)
SKIP_FORMAT_TOKENS = {"gguf", "onnx", "mlx", "openvino", "ov"}
VARIANT_MODIFIER_TOKENS = {"block", "dynamic", "static"}
VARIANT_TOKEN_PATTERN = re.compile(
    r"^(?:fp\d+(?:\.\d+)?|bf\d+(?:\.\d+)?|int\d+|uint\d+|nf\d+|mxfp\d+|nvfp\d+|w\d+a\d+|q\d+(?:_[a-z0-9]+)*|\d+bit)$",
    re.IGNORECASE,
)


def tokenize(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {token.lower() for token in re.findall(r"[A-Za-z]{4,}", text)}


def format_skip_reason(model_name: str, tags: set[str], file_names: list[str]) -> str:
    lowered_name = model_name.lower()
    if "gguf" in lowered_name or "gguf" in tags:
        return "gguf"
    if any(name.endswith(".gguf") for name in file_names):
        return "gguf"

    if "onnx" in lowered_name or "onnx" in tags:
        return "format"
    if any(
        name.endswith(".onnx") or name.startswith("onnx/") or "/onnx/" in name
        for name in file_names
    ):
        return "format"

    if re.search(r"(^|[-_])mlx($|[-_])", lowered_name) or "mlx" in tags:
        return "format"

    if "openvino" in lowered_name or "openvino" in tags:
        return "format"
    if re.search(r"(^|[-_])ov($|[-_])", lowered_name):
        return "format"
    if any("openvino" in name for name in file_names):
        return "format"

    return ""


def is_numeric_variant_suffix(suffix: str) -> bool:
    tokens = [token.lower() for token in suffix.split("-") if token]
    if not tokens:
        return False

    has_numeric_token = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in SKIP_FORMAT_TOKENS:
            return False
        if token in VARIANT_MODIFIER_TOKENS:
            index += 1
            continue
        if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] == "bit":
            has_numeric_token = True
            index += 2
            continue
        if VARIANT_TOKEN_PATTERN.fullmatch(token):
            has_numeric_token = True
            index += 1
            continue
        return False

    return has_numeric_token


def strip_numeric_variant_suffix(model_name: str) -> str:
    parts = model_name.split("-")
    for index in range(1, len(parts)):
        suffix = "-".join(parts[index:])
        if is_numeric_variant_suffix(suffix):
            return "-".join(parts[:index])

    return ""


def find_variant_path(
    org: str,
    model_name: str,
    tags: set[str],
    tracked_models_by_org: dict[str, list[tuple[str, str]]],
) -> str:
    base_tags: set[str] = set()
    for tag in tags:
        if not tag.startswith("base_model:"):
            continue
        base_model_id = tag.rsplit(":", 1)[-1]
        if "/" not in base_model_id:
            continue
        tag_org, base_name = base_model_id.split("/", 1)
        if tag_org == org:
            base_tags.add(base_name.lower())
            stem = strip_numeric_variant_suffix(base_name)
            if stem:
                base_tags.add(stem.lower())

    for base_name, csv_name in tracked_models_by_org[org]:
        if not model_name.startswith(base_name + "-"):
            continue
        if base_tags and base_name.lower() not in base_tags:
            continue
        if is_numeric_variant_suffix(model_name[len(base_name) + 1 :]):
            return csv_name

    return ""


def load_catalog() -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[str]],
    dict[str, datetime],
    dict[str, set[str]],
    set[str],
]:
    rows_by_path: dict[str, list[dict[str, str]]] = {}
    orgs_by_path: dict[str, list[str]] = {}
    cutoffs: dict[str, datetime] = {}
    tokens_by_org: dict[str, set[str]] = defaultdict(set)
    existing_model_ids: set[str] = set()

    for csv_name in CSV_NAMES:
        csv_path = REPO_ROOT / csv_name
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))

        rows_by_path[csv_name] = rows
        orgs_by_path[csv_name] = sorted({row["org"] for row in rows}, key=str.lower)

        for row in rows:
            existing_model_ids.add(row["modelId"])
            tokens_by_org[row["org"]].update(tokenize(row["model"]))

        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", csv_name],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        stamp = result.stdout.strip()
        if stamp:
            cutoffs[csv_name] = datetime.fromisoformat(stamp)
            continue

        cutoffs[csv_name] = datetime.fromtimestamp(
            csv_path.stat().st_mtime
        ).astimezone()

    # Shared orgs stay on the primary list so the two CSVs do not diverge.
    shared_orgs = set(orgs_by_path["models.csv"]) & set(
        orgs_by_path["extra_models.csv"]
    )
    orgs_by_path["extra_models.csv"] = [
        org for org in orgs_by_path["extra_models.csv"] if org not in shared_orgs
    ]
    return rows_by_path, orgs_by_path, cutoffs, tokens_by_org, existing_model_ids


def should_skip(model, org_tokens: set[str]) -> str:
    model_name = model.id.split("/", 1)[1]
    lowered_name = model_name.lower()
    if any(keyword in lowered_name for keyword in SKIP_KEYWORDS):
        return "keyword"
    if re.search(r"-seed\d+\b", lowered_name):
        return "keyword"
    if model.pipeline_tag in SKIP_PIPELINE_TAGS:
        return "pipeline"

    file_names = [sibling.rfilename.lower() for sibling in (model.siblings or [])]
    tags = {tag.lower() for tag in (model.tags or [])}
    format_reason = format_skip_reason(model_name, tags, file_names)
    if format_reason:
        return format_reason

    has_weights = any(name.endswith(WEIGHT_SUFFIXES) for name in file_names)
    has_hf_metadata = any(name.endswith(HF_METADATA_SUFFIXES) for name in file_names)

    if not has_weights:
        return "no_hf_files"
    if not has_hf_metadata:
        return "no_hf_files"

    if model.pipeline_tag is not None:
        return ""
    if tags & ALLOW_UNTAGGED_TAGS:
        return ""

    for token in tokenize(model_name):
        for existing_token in org_tokens:
            if token == existing_token:
                return ""
            if len(token) < 5 or len(existing_token) < 5:
                continue
            if token.startswith(existing_token) or existing_token.startswith(token):
                return ""

    return "untagged"


def write_rows(
    csv_path: Path, rows: list[dict[str, str]], additions: list[tuple[str, str, str]]
) -> None:
    merged_rows = rows + [
        {"org": org, "model": model_name, "modelId": model_id}
        for org, model_name, model_id in additions
    ]
    merged_rows.sort(
        key=lambda row: (
            row["org"].lower(),
            row["model"].lower(),
            row["modelId"].lower(),
        )
    )

    with csv_path.open("w", newline="\n") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["org", "model", "modelId"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(merged_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write", action="store_true", help="Update the CSV files in place."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print skipped model ids by reason."
    )
    args = parser.parse_args()

    rows_by_path, orgs_by_path, cutoffs, tokens_by_org, existing_model_ids = (
        load_catalog()
    )
    api = HfApi()
    additions_by_path: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    skipped_by_reason: dict[str, list[str]] = defaultdict(list)
    default_path_by_org: dict[str, str] = {}
    tracked_models_by_org: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for csv_name in CSV_NAMES:
        for org in orgs_by_path[csv_name]:
            default_path_by_org[org] = csv_name
        for row in rows_by_path[csv_name]:
            tracked_models_by_org[row["org"]].append((row["model"], csv_name))
            stem = strip_numeric_variant_suffix(row["model"])
            if stem:
                tracked_models_by_org[row["org"]].append((stem, csv_name))

    for tracked_models in tracked_models_by_org.values():
        tracked_models.sort(key=lambda row: (-len(row[0]), row[0].lower(), row[1]))

    for org in sorted(default_path_by_org, key=str.lower):
        default_path = default_path_by_org[org]
        for model in api.list_models(author=org, sort="created_at", full=True):
            if model.id in existing_model_ids:
                continue

            model_name = model.id.split("/", 1)[1]
            tags = {tag.lower() for tag in (model.tags or [])}
            variant_path = find_variant_path(
                org, model_name, tags, tracked_models_by_org
            )
            target_path = variant_path

            if not target_path:
                if not model.created_at or model.created_at <= cutoffs[default_path]:
                    continue
                target_path = default_path

            reason = should_skip(model, tokens_by_org[org])
            if reason:
                skipped_by_reason[reason].append(model.id)
                continue

            additions_by_path[target_path].append((org, model_name, model.id))
            existing_model_ids.add(model.id)
            tokens_by_org[org].update(tokenize(model_name))
            tracked_models_by_org[org].append((model_name, target_path))
            stem = strip_numeric_variant_suffix(model_name)
            if stem:
                tracked_models_by_org[org].append((stem, target_path))
            tracked_models_by_org[org].sort(
                key=lambda row: (-len(row[0]), row[0].lower(), row[1])
            )

    for csv_name in CSV_NAMES:
        additions = sorted(
            additions_by_path[csv_name],
            key=lambda row: (row[0].lower(), row[1].lower(), row[2].lower()),
        )
        print(
            f"{csv_name}: {len(additions)} additions since {cutoffs[csv_name].isoformat()}"
        )
        for _, _, model_id in additions:
            print(model_id)
        print()

        if not args.write or not additions:
            continue
        write_rows(REPO_ROOT / csv_name, rows_by_path[csv_name], additions)

    print("Skipped candidates:")
    for reason in sorted(skipped_by_reason):
        print(f"{reason}: {len(skipped_by_reason[reason])}")
        if not args.verbose:
            continue
        for model_id in skipped_by_reason[reason]:
            print(model_id)
        print()


if __name__ == "__main__":
    main()
