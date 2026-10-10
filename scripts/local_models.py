#!/usr/bin/env python3
"""Discover and validate a separate, evidence-based local model catalog.

The standard-library core accepts JSONL metadata. DuckDB is needed only when
reading Parquet. Canonical models.csv and extra_models.csv are always read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
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
COLUMNS = (
    "org",
    "model",
    "modelId",
    "parent_model_id",
    "parent_org",
    "parent_model",
    "parent_region",
    "parent_list",
    "local_formats",
    "base_model_id",
    "mapping_method",
    "already_tracked",
    "created_at",
    "variant_category",
    "is_modified",
    "modification_categories",
    "parent_lineage",
    "lineage_categories",
)
CATEGORIES = (
    "conversion",
    "finetune",
    "pruned",
    "adapter",
    "merge",
    "abliterated",
    "distilled",
    "unknown",
)
CHANGE_MARKERS = {
    "finetune": {
        "finetune",
        "finetuned",
        "sft",
        "dpo",
        "orpo",
        "grpo",
        "rlhf",
        "tuned",
    },
    "pruned": {"reap", "pruned", "pruning"},
    "adapter": {"lora", "adapter", "qlora"},
    "merge": {"merge", "merged"},
    "abliterated": {
        "abliterated",
        "abliteration",
        "obliterated",
        "heretic",
    },
    "distilled": {"distill", "distilled", "distillation"},
}
CHANGE_TAGS = {
    "reap": "pruned",
    "expert-pruning": "pruned",
    "pruned": "pruned",
    "pruning": "pruned",
    "abliteration": "abliterated",
    "abliterated": "abliterated",
    "heretic": "abliterated",
    "uncensored": "unknown",
}
ALIASES = {
    "gguf": "gguf",
    "mlx": "mlx",
    "mlx-lm": "mlx",
    "mlx-vlm": "mlx",
    "awq": "awq",
    "autoawq": "awq",
    "auto-awq": "awq",
    "gptq": "gptq",
    "auto-gptq": "gptq",
    "autogptq": "gptq",
    "auto_gptq": "gptq",
    "gptqmodel": "gptq",
    "exl2": "exl2",
    "onnx": "onnx",
    "onnxruntime": "onnx",
    "onnxruntime-web": "onnx",
    "onnxruntime-genai": "onnx",
    "openvino": "openvino",
    "hqq": "hqq",
    "exl3": "exl3",
    "exllamav3": "exl3",
    "exllama-v3": "exl3",
    "bitsandbytes": "bitsandbytes",
    "bits-and-bytes": "bitsandbytes",
    "bnb": "bitsandbytes",
    "aqlm": "aqlm",
    "quanto": "quanto",
    "optimum-quanto": "quanto",
    "ggml": "ggml",
    "compressed-tensors": "compressed-tensors",
    "auto-round": "auto-round",
    "autoround": "auto-round",
    "auto_round": "auto-round",
    "modelopt": "modelopt",
}
FORMAT_NAME_PATTERN = re.compile(
    r"(?<![a-z])(?:"
    + "|".join(re.escape(alias) for alias in sorted(ALIASES, key=len, reverse=True))
    + r")(?![a-z])"
)
NON_CONVERSION = {"finetune", "fine-tune", "adapter", "merge"}
TRAINING_WORDS = {
    "sft",
    "dpo",
    "abliterated",
    "uncensored",
    "finetune",
    "finetuned",
    "lora",
    "merge",
    "merged",
    "tuned",
    "obliterated",
    "distill",
    "distilled",
    "reap",
    "pruned",
    "pruning",
    "heretic",
}
CONVERSION_WORDS = set(ALIASES) | {
    "ggml",
    "llamacpp",
    "llama",
    "cpp",
    "quant",
    "quantized",
    "quantization",
    "quants",
    "quantisation",
    "converted",
    "conversion",
    "optimized",
    "optimised",
    "compatible",
    "weights",
    "safetensors",
    "fp",
    "bf",
    "f",
    "int",
    "q",
    "k",
    "m",
    "s",
    "l",
    "xl",
    "xxs",
    "xs",
    "iq",
    "i",
    "ud",
    "bpw",
    "bit",
    "bits",
    "g",
    "gs",
    "group",
    "size",
    "imatrix",
    "imat",
    "static",
    "dynamic",
    "hf",
    "cpu",
    "gpu",
    "cuda",
    "npu",
    "xpu",
    "ov",
    "web",
    "v",
    "model",
    "models",
    "exl",
    "exllama",
    "auto",
    "runtime",
    "lm",
    "vlm",
    "bnb",
    "nl",
    "w",
    "a",
    "x",
    "actorder",
    "true",
    "false",
    "desc",
    "act",
    "sym",
    "asym",
    "triton",
    "directml",
    "webgpu",
    "mxfp",
    "nvfp",
    "nf",
    "autoround",
    "round",
    "sharded",
}
REGIONS = {"us", "china", "eu", "other", "unknown"}


class CatalogError(ValueError):
    pass


def _repo(value: Any, field: str = "model ID") -> str:
    if not isinstance(value, str) or value != value.strip() or value.count("/") != 1:
        raise CatalogError(f"Invalid {field}: {value!r}")
    for part in value.split("/"):
        if (
            not re.fullmatch(r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?", part)
            or not any(char.isalnum() for char in part)
            or ".." in part
            or "--" in part
        ):
            raise CatalogError(f"Invalid {field}: {value!r}")
    return value


def _json_value(value: Any, expected: type, field: str):
    if value is None or value == "":
        return expected()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"Invalid JSON in {field}") from exc
    if value is None:
        return expected()
    if not isinstance(value, expected):
        raise CatalogError(f"Expected {expected.__name__} for {field}")
    return value


def _field(row: dict, plain: str, encoded: str, expected: type):
    return _json_value(row.get(plain, row.get(encoded)), expected, plain)


def creation_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CatalogError(
                f"Invalid metadata creation timestamp: {value!r}"
            ) from exc
    else:
        raise CatalogError("Invalid metadata creation timestamp")
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc)
    return stamp.date().isoformat()


def normalize_metadata(row: dict) -> dict:
    model_id = _repo(row.get("id", row.get("modelId")))
    tags = _field(row, "tags", "tags_json", list)
    bases = _field(row, "base_model_ids", "base_model_ids_json", list)
    if any(not isinstance(v, str) for v in tags):
        raise CatalogError(f"Non-string tags for {model_id}")
    bases = [value if isinstance(value, str) else repr(value) for value in bases]
    siblings = _field(row, "siblings", "siblings_json", list)
    files = []
    for item in siblings:
        value = (
            item.get("rfilename", item.get("path")) if isinstance(item, dict) else item
        )
        if not isinstance(value, str):
            raise CatalogError(f"Invalid sibling filename for {model_id}")
        files.append(value)
    for flag in ("private", "disabled"):
        if row.get(flag) is not None and type(row[flag]) is not bool:
            raise CatalogError(f"Invalid {flag} flag for {model_id}")
    config = _field(row, "config", "config_json", dict)
    card = _field(row, "card_data", "card_data_json", dict)
    file_counts = Counter(
        _file_format(filename) for filename in files if _file_format(filename)
    )
    representatives = {}
    for filename in sorted(set(files)):
        if fmt := _file_format(filename):
            representatives.setdefault(fmt, filename)
    return {
        "id": model_id,
        "library_name": (row.get("library_name") or "").casefold(),
        "tags": sorted(set(tags), key=lambda value: (value.casefold(), value)),
        "base_model_ids": sorted(
            set(bases), key=lambda value: (value.casefold(), value)
        ),
        "base_model_relation": row.get("base_model_relation") or None,
        "config": {
            key: config[key] for key in ("quantization_config",) if key in config
        },
        "card_data": {
            key: card[key]
            for key in ("base_model", "base_model_relation")
            if key in card
        },
        "siblings": sorted(representatives.values()),
        "file_counts": dict(file_counts),
        "private": bool(row.get("private")),
        "disabled": bool(row.get("disabled")),
        "created_at": creation_date(row.get("created_at")),
        "sha": row.get("sha"),
        "source_url": row.get("source_url"),
        "observed_at": row.get("observed_at"),
    }


def _file_format(filename: str) -> str | None:
    lower = filename.casefold()
    if lower.endswith(".gguf"):
        return "gguf"
    if lower.endswith(".onnx"):
        return "onnx"
    if re.search(r"(^|/)openvino[^/]*\.xml$", lower):
        return "openvino"
    if lower.endswith(".exl3"):
        return "exl3"
    if re.search(r"(^|/)ggml[^/]*\.bin$", lower):
        return "ggml"
    return None


def format_evidence(row: dict) -> list[dict]:
    evidence = []
    for kind, values in (("library", [row["library_name"]]), ("tag", row["tags"])):
        for value in values:
            if value.casefold() in ALIASES:
                evidence.append(
                    {"format": ALIASES[value.casefold()], "kind": kind, "value": value}
                )
    quant = row["config"].get("quantization_config", {})
    if isinstance(quant, dict):
        method = quant.get("quant_method", quant.get("quantization_method"))
        if isinstance(method, str) and method.casefold() in ALIASES:
            evidence.append(
                {
                    "format": ALIASES[method.casefold()],
                    "kind": "config",
                    "value": f"quantization_config.quant_method={method}",
                }
            )
    representatives = {}
    for filename in row["siblings"]:
        if fmt := _file_format(filename):
            representatives.setdefault(fmt, []).append(filename)
    for fmt, files in representatives.items():
        evidence.append(
            {
                "format": fmt,
                "kind": "file",
                "value": min(files),
                "matching_files": row.get("file_counts", {}).get(fmt, len(files)),
            }
        )
    unique = {(item["format"], item["kind"], item["value"]): item for item in evidence}
    return [
        unique[key]
        for key in sorted(
            unique, key=lambda item: (FORMATS.index(item[0]), item[1], item[2])
        )
    ]


def base_evidence(row: dict) -> dict:
    bases = list(row["base_model_ids"])
    metadata_relations = set()
    if row["base_model_relation"]:
        metadata_relations.add(str(row["base_model_relation"]).casefold())
    for tag in row["tags"]:
        if not tag.startswith("base_model:"):
            continue
        value = tag[len("base_model:") :]
        if ":" in value:
            relation, value = value.split(":", 1)
            metadata_relations.add(relation.casefold())
        if "/" in value:
            bases.append(value)
    card_base = row["card_data"].get("base_model", [])
    if card_base is None:
        card_base = []
    if isinstance(card_base, str):
        card_base = [card_base]
    if not isinstance(card_base, list):
        card_base = [repr(card_base)]
    card_base = [
        value if isinstance(value, str) else repr(value) for value in card_base
    ]
    bases.extend(card_base)
    explicit = row["card_data"].get("base_model_relation")
    if explicit is not None and not isinstance(explicit, str):
        raise CatalogError(f"Invalid card base_model_relation for {row['id']}")
    explicit = explicit.casefold() if explicit else None
    normalized = {}
    invalid = []
    for value in sorted(bases, key=lambda item: (item.casefold(), item)):
        try:
            parsed = _repo(value, "base model ID")
        except CatalogError:
            invalid.append(value)
            continue
        normalized.setdefault(parsed.casefold(), value)
    return {
        "base_model_ids": list(normalized.values()),
        "invalid_base_model_ids": sorted(set(invalid)),
        "metadata_relations": sorted(metadata_relations),
        "card_relation": explicit,
        "relation_provenance": "publisher_card"
        if explicit
        else "metadata_reported_or_unspecified",
        "modification_tags": sorted(
            {tag.casefold() for tag in row["tags"] if tag.casefold() in CHANGE_TAGS}
        ),
    }


def _compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def name_corroboration(model_id: str, base_id: str) -> tuple[bool, str]:
    """Require the checkpoint identity; only conversion modifiers may surround it."""
    name = _compact_name(model_id.split("/", 1)[1])
    base_org, base_name = base_id.split("/", 1)
    base = _compact_name(base_name)
    if not base or not name:
        return False, "invalid_checkpoint_name"
    # The explicit metadata edge still chooses the exact canonical repository.
    # These spelling alternatives only corroborate its checkpoint identity:
    # converters commonly omit the Transformers delivery suffix and Meta brand.
    delivery_name = re.sub(r"[._-]hf$", "", base_name, flags=re.IGNORECASE)
    if base_org.casefold() == "meta-llama":
        delivery_name = re.sub(r"^meta[._-]", "", delivery_name, flags=re.IGNORECASE)
    delivery_compact = _compact_name(delivery_name)
    used_alias = (
        bool(delivery_compact) and base not in name and delivery_compact in name
    )
    if used_alias:
        base = _compact_name(delivery_name)
    if base not in name:
        if any(word in name for word in TRAINING_WORDS):
            return False, "training_modifier_outside_parent"
        return False, "parent_name_not_preserved"
    # Remove the exact parent before looking for training modifiers: a tracked
    # distilled/finetuned checkpoint remains a legitimate original parent.
    remainder = name.replace(base, "", 1)
    if any(word in remainder for word in TRAINING_WORDS):
        return False, "training_modifier_outside_parent"
    # Work with original separators while stripping the parent's normalized
    # span, so Q4_K_M and publisher prefixes are recognizable modifiers.
    original = model_id.split("/", 1)[1].casefold()
    compact_positions = [
        index
        for index, char in enumerate(original)
        if char.isalnum() and char.isascii()
    ]
    position = name.index(base)
    before = original[: compact_positions[position]]
    after = original[compact_positions[position + len(base) - 1] + 1 :]
    tokens = re.findall(r"[a-z]+|[0-9]+", before + " " + after)
    allowed = (
        CONVERSION_WORDS
        | set(re.findall(r"[a-z]+|[0-9]+", model_id.split("/")[0].casefold()))
        | set(re.findall(r"[a-z]+|[0-9]+", base_id.split("/")[0].casefold()))
    )
    if any(not token.isdigit() and token not in allowed for token in tokens):
        return False, "unreviewed_name_modifier"
    return True, (
        "parent_name_with_delivery_alias"
        if used_alias
        else "parent_name_and_conversion_modifiers"
    )


def load_overrides(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise CatalogError("Overrides must be a JSON list")
    return validate_overrides(value)


def validate_overrides(overrides: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for original in overrides:
        row = dict(original)
        model_id = _repo(row.get("model_id"), "override model_id")
        if model_id.casefold() in seen:
            raise CatalogError(f"Duplicate override for {model_id}")
        seen.add(model_id.casefold())
        if row.get("action") not in {"include", "exclude"}:
            raise CatalogError(f"Invalid override action for {model_id}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise CatalogError(f"Override requires a review reason: {model_id}")
        urls = row.get("source_urls")
        if (
            not isinstance(urls, list)
            or not urls
            or any(
                not isinstance(url, str)
                or urlparse(url).scheme != "https"
                or not urlparse(url).netloc
                for url in urls
            )
        ):
            raise CatalogError(f"Override requires HTTPS evidence URLs: {model_id}")
        row["source_urls"] = sorted(set(urls))
        if row["action"] == "include":
            parent_id = _repo(row.get("parent_model_id"), "override parent_model_id")
            if parent_id.casefold() == model_id.casefold():
                raise CatalogError("Reviewed include cannot map a repository to itself")
            if "base_model_id" in row:
                base_id = _repo(row["base_model_id"], "override base_model_id")
                if base_id.casefold() == model_id.casefold():
                    raise CatalogError(
                        "Reviewed include cannot use itself as immediate base"
                    )
        if "local_formats" in row:
            formats = row["local_formats"]
            if (
                not isinstance(formats, list)
                or not formats
                or any(fmt not in FORMATS for fmt in formats)
            ):
                raise CatalogError(f"Invalid override formats for {model_id}")
            row["local_formats"] = [fmt for fmt in FORMATS if fmt in formats]
        categories = row.get("modification_categories", [])
        if not isinstance(categories, list) or any(
            category not in CATEGORIES[1:] for category in categories
        ):
            raise CatalogError(f"Invalid reviewed modification categories: {model_id}")
        if "modification_categories" in row:
            row["modification_categories"] = [
                category for category in CATEGORIES[1:] if category in categories
            ]
        result.append(row)
    return sorted(result, key=lambda row: row["model_id"].casefold())


def validate_aliases(rows: list[dict], parents: dict) -> dict:
    result = {}
    for original in rows:
        row = dict(original)
        alias = _repo(row.get("alias_id"), "alias_id")
        canonical = _repo(row.get("canonical_id"), "canonical_id")
        if (
            alias.casefold() in result
            or canonical.casefold() not in parents
            or alias.casefold() == canonical.casefold()
        ):
            raise CatalogError("Invalid or duplicate canonical repository alias")
        if (
            row.get("http_status") not in (301, 302, 307, 308)
            or row.get("redirect_url")
            != "https://huggingface.co/api/models/" + canonical
            or not re.fullmatch(r"[0-9a-f]{40}", row.get("resolved_sha", ""))
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
            or not isinstance(row.get("source_urls"), list)
            or "https://huggingface.co/api/models/" + alias not in row["source_urls"]
            or not row.get("observed_at")
        ):
            raise CatalogError(
                "Repository aliases require reviewed HTTP redirect evidence"
            )
        row["canonical_id"] = parents[canonical.casefold()]["modelId"]
        result[alias.casefold()] = row
    return result


def alias_base_evidence(base: dict, aliases: dict) -> dict:
    declared = base.get("declared_base_model_ids", base["base_model_ids"])
    resolved = {}
    proof = []
    for model_id in declared:
        alias = aliases.get(model_id.casefold())
        canonical = alias["canonical_id"] if alias else model_id
        resolved.setdefault(canonical.casefold(), canonical)
        if alias:
            proof.append(alias)
    return {
        **base,
        "declared_base_model_ids": declared,
        "base_model_ids": [resolved[key] for key in sorted(resolved)],
        "alias_evidence": sorted(proof, key=lambda row: row["alias_id"].casefold()),
    }


def edge_categories(
    step_id: str, next_id: str, root_id: str, step: dict, override: dict | None = None
) -> list[str]:
    """Classify changes relative to the immediate base; stop at the tracked root."""
    if override and override["action"] == "include":
        return override.get("modification_categories", []) or ["conversion"]
    base = step["base_evidence"]
    relations = set(base.get("metadata_relations", []))
    if base.get("card_relation"):
        relations.add(base["card_relation"])
    names = set(re.findall(r"[a-z]+", step_id.split("/")[1].casefold()))
    inherited_names = set(re.findall(r"[a-z]+", next_id.split("/")[1].casefold()))
    uncertain_behavior = "uncensored" in names - inherited_names
    categories = {
        category
        for category, markers in CHANGE_MARKERS.items()
        if (names - inherited_names) & markers
    }
    inherited_tags = set(step.get("base_modification_tags", []))
    for tag in set(base.get("modification_tags", [])) - inherited_tags:
        category = CHANGE_TAGS[tag]
        if category == "unknown":
            uncertain_behavior |= tag not in inherited_names
            continue
        if not inherited_names & CHANGE_MARKERS[category]:
            categories.add(category)
    relation_categories = {
        "finetune": "finetune",
        "fine-tune": "finetune",
        "adapter": "adapter",
        "merge": "merge",
        "pruned": "pruned",
        "pruning": "pruned",
        "distilled": "distilled",
        "distillation": "distilled",
        "abliterated": "abliterated",
    }
    categories.update(
        relation_categories[relation]
        for relation in relations
        if relation in relation_categories
    )
    if relations - set(relation_categories) - {"quantized", "conversion"}:
        categories.add("unknown")
    if not categories and uncertain_behavior:
        categories.add("unknown")
    if not categories:
        corroborated = (
            name_corroboration(step_id, next_id)[0]
            or name_corroboration(step_id, root_id)[0]
        )
        if step.get("format_evidence") and (
            corroborated or base.get("card_relation") == "quantized"
        ):
            return ["conversion"]
        return ["unknown"]
    return [category for category in CATEGORIES[1:] if category in categories]


def category_fields(
    chain: list[str], steps: list[dict], root_id: str, overrides: dict
) -> dict:
    edges = [
        edge_categories(
            chain[index],
            chain[index + 1],
            root_id,
            step,
            overrides.get(chain[index].casefold()),
        )
        for index, step in enumerate(steps)
    ]
    changes = [
        category
        for category in CATEGORIES[1:]
        if any(category in edge for edge in edges)
    ]
    return {
        "variant_category": "mixed"
        if len(changes) > 1
        else changes[0]
        if changes
        else "conversion",
        "is_modified": "true" if changes else "false",
        "modification_categories": ";".join(changes),
        "parent_lineage": ";".join(chain),
        "lineage_categories": ";".join("+".join(edge) for edge in edges),
    }


def _parent_index(parent_rows: list[dict]) -> dict:
    result = {}
    for original in parent_rows:
        row = dict(original)
        model_id = _repo(row.get("modelId"))
        org, model = model_id.split("/", 1)
        if (
            row.get("org") != org
            or row.get("model") != model
            or row.get("parent_list") not in {"models", "extra_models"}
        ):
            raise CatalogError(f"Invalid canonical parent row: {model_id}")
        key = model_id.casefold()
        if key in result:
            raise CatalogError(f"Duplicate canonical parent: {model_id}")
        result[key] = row
    return result


def discover_local_models(
    metadata_rows: Iterable[dict],
    parent_rows: list[dict],
    regions: dict[str, str],
    overrides: list[dict] | None = None,
    aliases: list[dict] | None = None,
) -> dict:
    """Resolve single-parent local descendant lineage to the nearest tracked model."""
    parents = _parent_index(parent_rows)
    alias_map = validate_aliases(aliases or [], parents)
    regions = {key.casefold(): value for key, value in regions.items()}
    if any(value not in REGIONS for value in regions.values()):
        raise CatalogError("Invalid parent region registry")
    override_map = {
        row["model_id"].casefold(): row for row in validate_overrides(overrides or [])
    }
    metadata = {}
    facts = {}
    for raw in metadata_rows:
        normalized = normalize_metadata(raw)
        key = normalized["id"].casefold()
        fact = {
            "formats": format_evidence(normalized),
            "base": alias_base_evidence(base_evidence(normalized), alias_map),
        }
        row = {
            field: normalized[field]
            for field in (
                "id",
                "library_name",
                "private",
                "disabled",
                "created_at",
                "sha",
                "source_url",
                "observed_at",
            )
        }
        if key in metadata and (metadata[key] != row or facts[key] != fact):
            raise CatalogError(f"Conflicting metadata rows: {row['id']}")
        metadata[key], facts[key] = row, fact
    for key, override in override_map.items():
        if key not in metadata:
            raise CatalogError(
                f"Override repository missing from metadata: {override['model_id']}"
            )
        if (
            override["action"] == "include"
            and override["parent_model_id"].casefold() not in parents
        ):
            raise CatalogError(
                f"Override parent is not canonical: {override['parent_model_id']}"
            )

    def tracked_conversion(key: str) -> bool:
        if key not in metadata:
            return False
        fact = facts[key]
        bases = fact["base"]["base_model_ids"]
        if len(bases) != 1:
            return False
        base_id = bases[0]
        step = {
            "base_evidence": fact["base"],
            "format_evidence": fact["formats"],
            "base_modification_tags": facts.get(base_id.casefold(), {})
            .get("base", {})
            .get("modification_tags", []),
        }
        if edge_categories(metadata[key]["id"], base_id, base_id, step) != [
            "conversion"
        ]:
            return False
        relations = set(fact["base"]["metadata_relations"]) | {
            fact["base"]["card_relation"]
        }
        name = metadata[key]["id"].split("/", 1)[1].casefold()
        return (
            bool(fact["formats"])
            and "quantized" in relations
            and any(
                re.search(rf"(^|[^a-z]){re.escape(item['format'])}([^a-z]|$)", name)
                for item in fact["formats"]
            )
        )

    memo = {}

    def resolve(
        key: str, visiting: tuple[str, ...] = ()
    ) -> tuple[str | None, str, list[str]]:
        if len(visiting) >= 64:
            return None, "lineage_depth_exceeded", [key]
        if key in visiting:
            return None, "lineage_cycle", list(visiting) + [key]
        if key in memo:
            return memo[key]
        if key not in metadata:
            # Canonical checkpoint identity is authoritative even when its
            # current Hub metadata is absent; no inferred namespace aliasing.
            return (
                (key, "canonical_parent", [key])
                if key in parents
                else (None, "missing_parent_metadata", [key])
            )
        row, fact = metadata[key], facts[key]
        if row["private"] or row["disabled"]:
            return None, "ineligible_repository", [key]
        if key in parents and not tracked_conversion(key):
            return key, "canonical_parent", [key]
        override = override_map.get(key)
        if override and override["action"] == "exclude":
            return None, "reviewed_exclusion", [key]
        if override and override["action"] == "include":
            parent = override["parent_model_id"].casefold()
            base_key = override.get(
                "base_model_id", override["parent_model_id"]
            ).casefold()
            if base_key == parent:
                return parent, "reviewed_override", [key, parent]
            resolved, reason, chain = resolve(base_key, (*visiting, key))
            if resolved != parent:
                raise CatalogError(
                    f"Reviewed immediate base does not reach declared parent: {row['id']} ({reason})"
                )
            return parent, "reviewed_override", [key, *chain]
        base = fact["base"]
        if base["invalid_base_model_ids"]:
            return None, "invalid_parent_evidence", [key]
        if len(base["base_model_ids"]) != 1:
            return (
                None,
                "ambiguous_parent"
                if base["base_model_ids"]
                else "missing_parent_evidence",
                [key],
            )
        base_id = base["base_model_ids"][0]
        parent, inherited, chain = resolve(base_id.casefold(), (*visiting, key))
        if parent is None:
            return None, inherited, [key, *chain]
        result = (
            parent,
            inherited
            if parent is None
            else "lineage_chain"
            if len(chain) > 1
            else "direct",
            [key, *chain],
        )
        memo[key] = result
        return result

    parent_name_prefixes = {}
    for parent in parents.values():
        name = _compact_name(parent["model"])
        width = min(8, len(name))
        parent_name_prefixes.setdefault(width, {}).setdefault(name[:width], []).append(
            name
        )

    def name_association(model_id: str) -> bool:
        name = _compact_name(model_id.split("/", 1)[1])
        for width, prefixes in parent_name_prefixes.items():
            for index in range(max(0, len(name) - width + 1)):
                if any(
                    name.startswith(parent, index)
                    for parent in prefixes.get(name[index : index + width], ())
                ):
                    return True
        return False

    rows, evidence, candidates = [], [], []
    for key in sorted(metadata):
        row, fact = metadata[key], facts[key]
        override = override_map.get(key)
        format_ids = [
            fmt
            for fmt in FORMATS
            if any(item["format"] == fmt for item in fact["formats"])
        ]
        if override and override.get("local_formats"):
            format_ids = override["local_formats"]
        name_hint = bool(
            FORMAT_NAME_PATTERN.search(row["id"].split("/", 1)[1].casefold())
        )
        if not format_ids and not override and not name_hint:
            continue
        parent_key, method, chain = resolve(key)
        reason = None
        if override and override["action"] == "exclude":
            reason = "reviewed_exclusion"
        elif not format_ids:
            reason = "name_only_format_hint" if name_hint else "missing_format_evidence"
        elif parent_key == key:
            reason = "tracked_original"
        elif parent_key is None:
            reason = method
        if reason:
            associated = (
                any(item in parents for item in chain)
                or any(
                    base.casefold() in parents
                    for base in fact["base"]["base_model_ids"]
                )
                or name_association(row["id"])
            )
            candidates.append(
                {
                    "tracked_association": associated,
                    "model_id": row["id"],
                    "publisher": row["id"].split("/")[0],
                    "local_formats": format_ids,
                    "reason": reason,
                    "base_model_ids": fact["base"]["base_model_ids"],
                    "format_evidence": fact["formats"],
                    "base_evidence": fact["base"],
                    "lineage": [
                        metadata[item]["id"]
                        if item in metadata
                        else parents[item]["modelId"]
                        if item in parents
                        else item
                        for item in chain
                    ],
                    "override": override,
                }
            )
            continue
        parent = parents[parent_key]
        org, model = row["id"].split("/", 1)
        bases = fact["base"]["base_model_ids"]
        direct = bases[0] if len(bases) == 1 else ""
        if direct.casefold() in metadata:
            direct = metadata[direct.casefold()]["id"]
        elif direct.casefold() in parents:
            direct = parents[direct.casefold()]["modelId"]
        if override and override["action"] == "include":
            method = "reviewed_override"
            immediate = chain[1]
            direct = (
                parents[immediate]["modelId"]
                if immediate in parents
                else metadata[immediate]["id"]
            )
        output = {
            "org": org,
            "model": model,
            "modelId": row["id"],
            "parent_model_id": parent["modelId"],
            "parent_org": parent["org"],
            "parent_model": parent["model"],
            "parent_region": regions.get(parent["org"].casefold(), "unknown"),
            "parent_list": parent["parent_list"],
            "local_formats": ";".join(format_ids),
            "base_model_id": direct,
            "mapping_method": method,
            "already_tracked": "true" if key in parents else "false",
            "created_at": row["created_at"],
        }
        rows.append(output)
        evidence.append(
            {
                "model_id": row["id"],
                "parent_model_id": parent["modelId"],
                "mapping_method": method,
                "local_formats": format_ids,
                "format_evidence": fact["formats"],
                "base_evidence": fact["base"],
                "base_modification_tags": facts.get(chain[1], {})
                .get("base", {})
                .get("modification_tags", []),
                "lineage": [
                    metadata[item]["id"]
                    if item in metadata
                    else parents[item]["modelId"]
                    if item in parents
                    else item
                    for item in chain
                ],
                "name_corroboration": name_corroboration(row["id"], parent["modelId"]),
                "intermediate_evidence": [
                    {
                        "model_id": metadata[item]["id"],
                        "format_evidence": facts[item]["formats"],
                        "base_evidence": facts[item]["base"],
                        "base_modification_tags": facts.get(chain[index + 2], {})
                        .get("base", {})
                        .get("modification_tags", []),
                        "override": override_map.get(item),
                        "private": metadata[item]["private"],
                        "disabled": metadata[item]["disabled"],
                    }
                    for index, item in enumerate(chain[1:-1])
                    if item in metadata
                ],
                "metadata": {
                    "library_name": row["library_name"],
                    "created_at": row["created_at"],
                    "private": row["private"],
                    "disabled": row["disabled"],
                    "sha": row["sha"],
                    "source_url": row["source_url"],
                    "observed_at": row["observed_at"],
                },
                "override": override,
            }
        )
        fields = category_fields(
            evidence[-1]["lineage"],
            [evidence[-1], *evidence[-1]["intermediate_evidence"]],
            parent["modelId"],
            override_map,
        )
        output.update(fields)
        evidence[-1].update(fields)
    covered_parents = {row["parent_model_id"] for row in rows}
    report = {
        "schema_version": 2,
        "counts": {
            "metadata_repositories": len(metadata),
            "candidate_repositories": len(rows) + len(candidates),
            "included_repositories": len(rows),
            "unresolved_repositories": len(candidates),
            "metadata_without_local_signals": len(metadata)
            - len(rows)
            - len(candidates),
        },
        "candidate_definition": "format_evidence_or_local_format_name_hint_or_reviewed_override",
        "included_by_format": dict(
            sorted(
                Counter(
                    fmt for row in rows for fmt in row["local_formats"].split(";")
                ).items()
            )
        ),
        "included_by_parent": dict(
            sorted(
                Counter(row["parent_model_id"] for row in rows).items(),
                key=lambda item: item[0].casefold(),
            )
        ),
        "included_by_parent_list": dict(
            sorted(Counter(row["parent_list"] for row in rows).items())
        ),
        "included_by_category": dict(
            sorted(Counter(row["variant_category"] for row in rows).items())
        ),
        "included_by_modified": dict(
            sorted(Counter(row["is_modified"] for row in rows).items())
        ),
        "unresolved_by_reason": dict(
            sorted(Counter(row["reason"] for row in candidates).items())
        ),
        "unresolved_by_parent": dict(
            sorted(
                Counter(
                    parent for row in candidates for parent in row["base_model_ids"]
                ).items(),
                key=lambda item: item[0].casefold(),
            )
        ),
        "unresolved_by_publisher": dict(
            sorted(
                Counter(row["publisher"] for row in candidates).items(),
                key=lambda item: (-item[1], item[0].casefold()),
            )[:100]
        ),
        "reviewed_overrides": len(override_map),
        "parent_coverage": {
            label: {
                "tracked_parents": sum(
                    parent["parent_list"] == label for parent in parents.values()
                ),
                "parents_with_observed_variants": len(
                    {
                        row["parent_model_id"]
                        for row in rows
                        if row["parent_list"] == label
                    }
                ),
                "parents_without_observed_variants": sorted(
                    [
                        parent["modelId"]
                        for parent in parents.values()
                        if parent["parent_list"] == label
                        and parent["modelId"] not in covered_parents
                    ],
                    key=str.casefold,
                ),
            }
            for label in ("models", "extra_models")
        },
    }
    return {
        "rows": rows,
        "evidence": evidence,
        "candidates": candidates,
        "report": report,
    }


def load_parents(models_path: Path, extra_path: Path) -> list[dict]:
    result = []
    for path, label in ((models_path, "models"), (extra_path, "extra_models")):
        reader = csv.DictReader(io.StringIO(path.read_text()))
        if reader.fieldnames != ["org", "model", "modelId"]:
            raise CatalogError(f"Invalid canonical model CSV header: {path}")
        result.extend({**row, "parent_list": label} for row in reader)
    _parent_index(result)
    return result


def load_regions(path: Path) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(path.read_text()))
    if reader.fieldnames != ["hf_org", "region", "notes"]:
        raise CatalogError("Invalid organization region header")
    result = {}
    for row in reader:
        org, region, notes = row.get("hf_org"), row.get("region"), row.get("notes")
        if (
            None in row
            or not org
            or org != org.strip()
            or region not in REGIONS
            or notes is None
            or notes != notes.strip()
            or (region == "unknown" and not notes)
            or org.casefold() in result
        ):
            raise CatalogError("Invalid organization region row")
        result[org.casefold()] = region
    return result


def iter_metadata(path: Path) -> Iterable[dict]:
    if path.suffix.casefold() not in {".parquet", ".pq"}:
        with path.open() as handle:
            for index, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise CatalogError(f"Metadata row {index} must be an object")
                    yield row
        return
    try:
        import duckdb
    except ImportError as exc:
        raise CatalogError(
            "Parquet input requires optional duckdb; JSONL needs only Python"
        ) from exc
    connection = duckdb.connect()
    try:
        names = {
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
        }
        wanted = [
            key
            for key in (
                "id",
                "library_name",
                "tags",
                "tags_json",
                "base_model_relation",
                "base_model_ids",
                "base_model_ids_json",
                "config",
                "config_json",
                "card_data",
                "card_data_json",
                "siblings",
                "siblings_json",
                "created_at",
                "private",
                "disabled",
                "sha",
                "source_url",
                "observed_at",
            )
            if key in names
        ]
        if "id" not in wanted:
            raise CatalogError("Parquet metadata requires id column")
        cursor = connection.execute(
            "SELECT "
            + ", ".join('"' + key + '"' for key in wanted)
            + " FROM read_parquet(?)",
            [str(path)],
        )
        while batch := cursor.fetchmany(5000):
            for row in batch:
                yield dict(zip(wanted, row))
    finally:
        connection.close()


def csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def candidate_csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = ["model_id", "local_formats", "reason", "base_model_ids", "lineage"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: ";".join(row[key]) if isinstance(row[key], list) else row[key]
                for key in fields
            }
        )
    return buffer.getvalue().encode()


def jsonl_bytes(rows: list[dict], *, compressed: bool = False) -> bytes:
    payload = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode()
        for row in rows
    )
    if not compressed:
        return payload
    buffer = io.BytesIO()
    # An empty filename and fixed mtime avoid machine/time-dependent headers.
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_manifest(path: Path, metadata_sha256: str, rows: int) -> str:
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("sha256") != metadata_sha256
        or type(manifest.get("rows")) is not int
        or manifest["rows"] != rows
    ):
        raise CatalogError("Source manifest does not match metadata hash and row count")
    return _sha(payload)


def validate_catalog_rows(
    rows: list[dict],
    parent_rows: list[dict],
    regions: dict,
    overrides: list[dict] | None = None,
    aliases: list[dict] | None = None,
) -> None:
    """Check CSV consistency and reviewed corrections, not unavailable source proof."""
    parents = _parent_index(parent_rows)
    validate_aliases(aliases or [], parents)
    override_map = {
        row["model_id"].casefold(): row for row in validate_overrides(overrides or [])
    }
    for row in rows:
        if set(row) != set(COLUMNS) or any(
            not isinstance(value, str) for value in row.values()
        ):
            raise CatalogError("Invalid local catalog row shape")
    if [row["modelId"].casefold() for row in rows] != sorted(
        row["modelId"].casefold() for row in rows
    ):
        raise CatalogError("Catalog must be sorted by case-folded modelId")
    seen = set()
    for row in rows:
        key = _repo(row["modelId"]).casefold()
        if key in seen:
            raise CatalogError(f"Duplicate catalog repository: {row['modelId']}")
        seen.add(key)
        if row["org"] + "/" + row["model"] != row["modelId"]:
            raise CatalogError("Catalog repository columns disagree")
        parent = parents.get(row["parent_model_id"].casefold())
        if parent is None or row["parent_model_id"] != parent["modelId"]:
            raise CatalogError("Catalog parent is missing or noncanonical")
        for field, expected in (
            ("parent_org", parent["org"]),
            ("parent_model", parent["model"]),
            ("parent_list", parent["parent_list"]),
            ("parent_region", regions.get(parent["org"].casefold(), "unknown")),
        ):
            if row[field] != expected:
                raise CatalogError(f"Catalog {field} disagrees with canonical parent")
        formats = row["local_formats"].split(";")
        if not formats or formats != [fmt for fmt in FORMATS if fmt in formats]:
            raise CatalogError("Invalid, duplicate, or unsorted local formats")
        if row["already_tracked"] != ("true" if key in parents else "false"):
            raise CatalogError("Catalog tracked membership disagrees")
        if row["created_at"] and creation_date(row["created_at"]) != row["created_at"]:
            raise CatalogError("Invalid creation date")
        if row["mapping_method"] not in {
            "direct",
            "lineage_chain",
            "reviewed_override",
        }:
            raise CatalogError("Invalid mapping method")
        _repo(row["base_model_id"], "base_model_id")
        chain = row["parent_lineage"].split(";")
        for part in chain:
            _repo(part, "lineage repository")
        if (
            len(chain) < 2
            or chain[0] != row["modelId"]
            or chain[-1] != row["parent_model_id"]
            or len({part.casefold() for part in chain}) != len(chain)
            or row["base_model_id"] != chain[1]
        ):
            raise CatalogError("Invalid catalog lineage or immediate base")
        if row["mapping_method"] == "direct" and len(chain) != 2:
            raise CatalogError("Direct mapping has intermediate lineage")
        if row["mapping_method"] == "lineage_chain" and len(chain) < 3:
            raise CatalogError("Lineage chain lacks an intermediate")
        edges = [edge.split("+") for edge in row["lineage_categories"].split(";")]
        if len(edges) != len(chain) - 1 or any(
            edge != ["conversion"]
            and (not edge or edge != [c for c in CATEGORIES[1:] if c in edge])
            for edge in edges
        ):
            raise CatalogError("Invalid catalog lineage categories")
        changes = [c for c in CATEGORIES[1:] if any(c in edge for edge in edges)]
        expected = {
            "variant_category": "mixed"
            if len(changes) > 1
            else changes[0]
            if changes
            else "conversion",
            "is_modified": "true" if changes else "false",
            "modification_categories": ";".join(changes),
        }
        if any(row[field] != value for field, value in expected.items()):
            raise CatalogError("Catalog modification categories disagree with lineage")
        leaf_override = override_map.get(key)
        if (row["mapping_method"] == "reviewed_override") != bool(
            leaf_override and leaf_override["action"] == "include"
        ):
            raise CatalogError("Reviewed mapping lacks matching override")
        for index, step_id in enumerate(chain[:-1]):
            reviewed = override_map.get(step_id.casefold())
            if not reviewed:
                continue
            if reviewed["action"] == "exclude":
                raise CatalogError("Reviewed exclusion appears in catalog lineage")
            if (
                reviewed["parent_model_id"].casefold() != chain[-1].casefold()
                or reviewed.get("base_model_id", reviewed["parent_model_id"]).casefold()
                != chain[index + 1].casefold()
                or edges[index]
                != (reviewed.get("modification_categories", []) or ["conversion"])
            ):
                raise CatalogError(
                    "Reviewed lineage disagrees with corrected base, parent, or categories"
                )
            if (
                index == 0
                and reviewed.get("local_formats")
                and formats != reviewed["local_formats"]
            ):
                raise CatalogError("Local formats disagree with reviewed override")


def validate_catalog(
    rows: list[dict],
    evidence: list[dict],
    parent_rows: list[dict],
    regions: dict,
    overrides: list[dict] | None = None,
    aliases: list[dict] | None = None,
) -> None:
    """Validate the CSV and reconstruct its claims from source evidence."""
    validate_catalog_rows(rows, parent_rows, regions, overrides, aliases)
    parents = _parent_index(parent_rows)
    alias_map = validate_aliases(aliases or [], parents)
    override_map = {
        row["model_id"].casefold(): row for row in validate_overrides(overrides or [])
    }
    seen = {row["modelId"].casefold() for row in rows}
    facts = {}
    for item in evidence:
        key = _repo(item.get("model_id"), "evidence model_id").casefold()
        if key in facts:
            raise CatalogError("Duplicate evidence repository")
        facts[key] = item
    for row in rows:
        key = row["modelId"].casefold()
        formats = row["local_formats"].split(";")
        item = facts.get(key)
        if (
            item is None
            or item.get("parent_model_id") != row["parent_model_id"]
            or item.get("mapping_method") != row["mapping_method"]
            or item.get("local_formats") != formats
        ):
            raise CatalogError("Catalog and evidence disagree")
        chain = row["parent_lineage"].split(";")
        if item.get("lineage") != chain:
            raise CatalogError("Catalog and evidence lineage disagree")
        if item.get("override") != override_map.get(key):
            raise CatalogError("Evidence override does not match reviewed source")

        def check_step(step, step_id, next_id, expected_formats=None):
            reviewed = override_map.get(step_id.casefold())
            if reviewed and reviewed["action"] == "exclude":
                raise CatalogError("Reviewed exclusion appears in catalog lineage")
            if step.get("override") != reviewed:
                raise CatalogError("Lineage review evidence disagrees with overrides")
            proof = step.get("format_evidence")
            if not isinstance(proof, list):
                raise CatalogError("Invalid format evidence")
            proved = set()
            for signal in proof:
                kind, value, fmt = (
                    signal.get("kind"),
                    signal.get("value"),
                    signal.get("format"),
                )
                if not isinstance(value, str) or fmt not in FORMATS:
                    raise CatalogError("Invalid format evidence signal")
                inferred = (
                    _file_format(value)
                    if kind == "file"
                    else ALIASES.get(value.casefold())
                    if kind in {"tag", "library"}
                    else ALIASES.get(value.split("=", 1)[-1].casefold())
                    if kind == "config"
                    else None
                )
                if inferred != fmt:
                    raise CatalogError("Format evidence signal does not support format")
                proved.add(fmt)
            if reviewed and reviewed.get("local_formats"):
                proved = set(reviewed["local_formats"])
            if expected_formats is not None and (
                not proved or proved != set(expected_formats)
            ):
                raise CatalogError("Local formats lack matching evidence")
            base = step.get("base_evidence", {})
            if alias_base_evidence(base, alias_map) != base:
                raise CatalogError(
                    "Lineage aliases disagree with reviewed redirect evidence"
                )
            if reviewed and reviewed["action"] == "include":
                if (
                    reviewed["parent_model_id"].casefold()
                    != row["parent_model_id"].casefold()
                    or reviewed.get(
                        "base_model_id", reviewed["parent_model_id"]
                    ).casefold()
                    != next_id.casefold()
                ):
                    raise CatalogError(
                        "Reviewed lineage disagrees with corrected base or parent"
                    )
            else:
                if (
                    base.get("invalid_base_model_ids")
                    or len(base.get("base_model_ids", [])) != 1
                    or base["base_model_ids"][0].casefold() != next_id.casefold()
                ):
                    raise CatalogError(
                        "Automatic lineage disagrees with source base evidence"
                    )

        if any(
            item.get("metadata", {}).get(flag) is not False
            for flag in ("private", "disabled")
        ):
            raise CatalogError("Included metadata eligibility is not evidenced")
        check_step(item, row["modelId"], chain[1], formats)
        intermediates = item.get("intermediate_evidence", [])
        if [step.get("model_id") for step in intermediates] != chain[1:-1]:
            raise CatalogError("Intermediate lineage evidence is incomplete")
        for index, step in enumerate(intermediates, 1):
            if any(step.get(flag) is not False for flag in ("private", "disabled")):
                raise CatalogError("Ineligible intermediate lineage")
            check_step(step, chain[index], chain[index + 1])
        categories = category_fields(
            chain, [item, *intermediates], row["parent_model_id"], override_map
        )
        if any(
            row[field] != value or item.get(field) != value
            for field, value in categories.items()
        ):
            raise CatalogError(
                "Catalog modification categories disagree with lineage evidence"
            )
    if seen != set(facts):
        raise CatalogError("Evidence must join one-to-one with catalog")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / "models.csv")
    parser.add_argument("--extra-models", type=Path, default=ROOT / "extra_models.csv")
    parser.add_argument(
        "--regions", type=Path, default=ROOT / "metadata/organization_regions.csv"
    )
    parser.add_argument(
        "--overrides", type=Path, default=ROOT / "metadata/local_models_overrides.json"
    )
    parser.add_argument(
        "--aliases", type=Path, default=ROOT / "metadata/local_models_aliases.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "local-models.csv")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / ".local-model-cache/local_models_evidence.jsonl.gz",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        help="Optional unresolved review queue with tracked-parent association",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Write complete unresolved audit (use outside the repository)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / ".local-model-cache/local_models_report.json",
    )
    parser.add_argument("--source-url")
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Public manifest with metadata sha256 and rows",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Replay metadata and compare exact artifact bytes",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Validate CSV structure and reviewed corrections without source evidence",
    )
    args = parser.parse_args(argv)
    parents = load_parents(args.models, args.extra_models)
    regions = load_regions(args.regions)
    overrides = load_overrides(args.overrides)
    aliases = json.loads(args.aliases.read_text()) if args.aliases.exists() else []
    if not isinstance(aliases, list):
        raise CatalogError("Repository aliases must be a JSON list")
    if args.validate:
        payload = args.output.read_bytes()
        reader = csv.DictReader(io.StringIO(payload.decode()))
        if reader.fieldnames != list(COLUMNS):
            raise CatalogError("Invalid catalog header")
        rows = list(reader)
        validate_catalog_rows(rows, parents, regions, overrides, aliases)
        if b"\r" in payload:
            raise CatalogError("Catalog must use Unix newlines")
        print(
            f"Validated {len(rows)} local repositories: CSV structure and reviewed corrections; source evidence not checked"
        )
        return
    input_paths = {
        "models": args.models,
        "extra_models": args.extra_models,
        "regions": args.regions,
    }
    if args.overrides and args.overrides.exists():
        input_paths["overrides"] = args.overrides
    if args.source_manifest:
        input_paths["source_manifest"] = args.source_manifest
    if args.aliases.exists():
        input_paths["aliases"] = args.aliases
    inputs = {key: _sha(path.read_bytes()) for key, path in input_paths.items()}
    output_paths = {
        "catalog": args.output,
        "evidence": args.evidence,
        "report": args.report,
    }
    if args.candidates:
        output_paths["candidates"] = args.candidates
    if len({path.resolve() for path in output_paths.values()}) != len(
        output_paths
    ) or set(path.resolve() for path in output_paths.values()) & set(
        path.resolve() for path in input_paths.values()
    ):
        raise CatalogError(
            "Output paths must be distinct and must not overwrite canonical input lists"
        )
    if args.metadata is None:
        parser.error("--metadata is required unless --validate is used")
    result = discover_local_models(
        iter_metadata(args.metadata), parents, regions, overrides, aliases
    )
    validate_catalog(
        result["rows"], result["evidence"], parents, regions, overrides, aliases
    )
    payloads = {
        "catalog": csv_bytes(result["rows"]),
        "evidence": jsonl_bytes(
            result["evidence"], compressed=args.evidence.suffix.casefold() == ".gz"
        ),
    }
    if args.candidates:
        review_rows = [
            row
            for row in result["candidates"]
            if args.all_candidates or row["tracked_association"]
        ]
        payloads["candidates"] = (
            candidate_csv_bytes(review_rows)
            if args.candidates.suffix.casefold() == ".csv"
            else jsonl_bytes(review_rows)
        )
    report = result["report"]
    report["candidate_scope"] = (
        "all_unresolved" if args.all_candidates else "tracked_parent_association"
    )
    report["review_queue_repositories"] = sum(
        args.all_candidates or row["tracked_association"]
        for row in result["candidates"]
    )
    source = {
        "metadata_sha256": file_sha(args.metadata),
        "url": args.source_url,
        "revision": args.source_revision,
    }
    if args.source_manifest:
        source["manifest_sha256"] = validate_source_manifest(
            args.source_manifest,
            source["metadata_sha256"],
            report["counts"]["metadata_repositories"],
        )
    report.update(
        inputs=inputs,
        source=source,
        artifacts={key: _sha(value) for key, value in payloads.items()},
    )
    payloads["report"] = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    for key, payload in payloads.items():
        path = output_paths[key]
        if args.check:
            if not path.exists() or path.read_bytes() != payload:
                raise CatalogError(f"Generated {key} differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(
        f"{'Checked' if args.check else 'Wrote'} {len(result['rows'])} local repositories; {len(result['candidates'])} unresolved candidates"
    )


if __name__ == "__main__":
    try:
        main()
    except (CatalogError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
