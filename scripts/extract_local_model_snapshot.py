#!/usr/bin/env python3
"""Extract public local-model discovery metadata and ancestors from a pinned Parquet.

The output is a candidate cache, not an approved catalog. Name-only and inferred
relations deliberately remain available for the separate catalog review policy.
Only DuckDB beyond the Python standard library is needed; this script makes no
network requests and never changes the canonical tracked lists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ALIASES = (
    "gguf", "ggml", "llama.cpp", "mlx", "mlx-lm", "mlx-vlm", "mlx_lm", "mlx_vlm",
    "awq", "autoawq", "auto-awq", "gptq", "autogptq", "auto-gptq", "gptqmodel",
    "exl2", "exllamav2", "onnx", "onnxruntime", "onnxruntime-web", "onnxruntime-genai",
    "onnxruntime_genai", "onnx-genai", "openvino", "openvino-genai", "openvino-export",
    "hqq", "exl3", "exllamav3", "bitsandbytes", "bnb", "bitsandbytes-4bit",
    "bitsandbytes-8bit", "compressed-tensors", "aqlm", "quanto", "optimum-quanto", "eetq", "auto-round", "autoround", "auto_round", "modelopt",
)
COLUMNS = (
    "id", "author", "created_at", "private", "disabled", "library_name", "pipeline_tag",
    "tags_json", "config_json", "card_data_json", "siblings_json",
    "base_model_relation", "base_model_ids_json",
)
CARD_FIELDS = frozenset({
    "base_model", "base_model_relation", "library_name", "pipeline_tag", "tags",
    "quantization_config", "quantization", "quantized_by",
})
ARTIFACT_PATTERN = r'\.gguf"|\.onnx"|openvino[^" ]*\.xml"|\.exl3"|\.ggml"|ggml[^" ]*\.bin"'
NAME_PATTERN = (
    r"(?:^|[._/ -])(?:gguf|ggml|mlx|awq|gptq|exl2|exl3|hqq|bnb|bitsandbytes|"
    r"openvino|onnx|aqlm|quanto|auto[-_]?round|modelopt|4bit|8bit|4-bit|8-bit|q[2-8]_[a-z0-9]+)(?:$|[._/ -])"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_parents(primary: Path, extra: Path) -> dict[str, dict]:
    parents: dict[str, dict] = {}
    for path, kind in ((primary, "models"), (extra, "extra_models")):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["org", "model", "modelId"]:
                raise ValueError(f"Unexpected tracked-list header in {path}")
            for row in reader:
                if any(not isinstance(row.get(key), str) for key in reader.fieldnames):
                    raise ValueError(f"Malformed tracked-list row in {path}")
                model_id = row["modelId"]
                if model_id != f"{row['org']}/{row['model']}" or model_id.strip() != model_id:
                    raise ValueError(f"Inconsistent tracked model id: {model_id}")
                key = model_id.casefold()
                if key in parents:
                    raise ValueError(f"Duplicate tracked parent id: {model_id}")
                parents[key] = {"model_id": model_id, "parent_list": kind}
    if not parents:
        raise ValueError("Tracked parent lists are empty")
    return parents


def _decoded(value: Any, expected: type, field: str) -> Any:
    if value is None or value == "":
        return expected()
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, expected):
        raise ValueError(f"Expected {expected.__name__} in {field}")
    return parsed


def normalize_metadata(row: dict) -> dict:
    if row.get("private"):
        raise ValueError("Private metadata cannot enter the public discovery cache")
    tags = _decoded(row.get("tags_json"), list, "tags_json")
    bases = _decoded(row.get("base_model_ids_json"), list, "base_model_ids_json")
    if any(not isinstance(value, str) for value in tags + bases):
        raise ValueError("Model tags and bases must be strings")
    card = _decoded(row.get("card_data_json"), dict, "card_data_json")
    siblings = _decoded(row.get("siblings_json"), list, "siblings_json")
    names = []
    for sibling in siblings:
        name = sibling if isinstance(sibling, str) else sibling.get("rfilename") if isinstance(sibling, dict) else None
        if not isinstance(name, str):
            raise ValueError("Invalid repository filename metadata")
        names.append(name)
    return {
        **{key: row.get(key) for key in COLUMNS if not key.endswith("_json")},
        "tags": tags,
        "config": _decoded(row.get("config_json"), dict, "config_json"),
        "card_data": {key: value for key, value in card.items() if key in CARD_FIELDS},
        "siblings": names,
        "base_model_ids": bases,
        "source_url": f"https://huggingface.co/{row['id']}",
    }


def parent_keys(row: dict) -> set[str]:
    bases = set(row["base_model_ids"])
    for tag in row["tags"]:
        if tag.startswith("base_model:"):
            value = tag[len("base_model:"):]
            if ":" in value:
                value = value.split(":", 1)[1]
            if "/" in value:
                bases.add(value)
    card_base = row["card_data"].get("base_model")
    if isinstance(card_base, str):
        bases.add(card_base)
    elif isinstance(card_base, list):
        bases.update(value for value in card_base if isinstance(value, str))
    return {value.casefold() for value in bases if "/" in value}


def extract_snapshot(
    input_parquet: Path, primary: Path, extra: Path, output: Path, manifest: Path,
    *, dataset_revision: str | None = None, metrics_parquet: Path | None = None,
    metrics_date: str | None = None, max_ancestor_depth: int = 16,
) -> dict:
    import duckdb

    inputs = {path.resolve() for path in (input_parquet, primary, extra)}
    if metrics_parquet is not None:
        inputs.add(metrics_parquet.resolve())
    outputs = {output.resolve(), manifest.resolve()}
    if len(outputs) != 2 or inputs & outputs:
        raise ValueError("Output and manifest must be distinct from each other and every input")
    parents = read_parents(primary, extra)
    if max_ancestor_depth < 1:
        raise ValueError("max_ancestor_depth must be positive")
    connection = duckdb.connect()
    connection.execute("SET memory_limit='2GB'")
    connection.execute("SET threads=2")
    connection.execute("CREATE TABLE parent_ids AS SELECT CAST(unnest(?) AS VARCHAR) AS id", [list(parents)])
    parent_names = sorted({value["model_id"].split("/", 1)[1].casefold() for value in parents.values()})
    parent_pattern = "(?:" + "|".join(re.escape(name) for name in parent_names) + ")"
    tag_pattern = '"(' + "|".join(re.escape(alias) for alias in ALIASES) + ')"'
    columns = ",".join(COLUMNS)
    total, public = connection.execute(
        "SELECT COUNT(*),COUNT(*) FILTER(WHERE private IS NOT TRUE) FROM read_parquet(?)",
        [str(input_parquet)],
    ).fetchone()
    print(f"Scanning {total:,} metadata rows for public discovery candidates", flush=True)
    cursor = connection.execute(f"""
        SELECT {columns} FROM read_parquet(?) WHERE private IS NOT TRUE AND (
            lower(id) IN (SELECT id FROM parent_ids)
            OR lower(library_name) IN (SELECT unnest(?))
            OR regexp_matches(lower(tags_json), ?)
            OR regexp_matches(lower(siblings_json), ?)
            OR contains(lower(config_json), 'quantization_config')
            OR contains(lower(card_data_json), 'quantization_config')
            OR regexp_matches(lower(id), ?)
            OR regexp_matches(lower(id), ?)
        )
    """, [str(input_parquet), list(ALIASES), tag_pattern, ARTIFACT_PATTERN,
           NAME_PATTERN, parent_pattern])
    models: dict[str, dict] = {}

    def consume_rows(cursor) -> set[str]:
        seen = set()
        names = [column[0] for column in cursor.description]
        while batch := cursor.fetchmany(5000):
            for values in batch:
                row = normalize_metadata(dict(zip(names, values)))
                key = row["id"].casefold()
                if key in models:
                    raise ValueError(f"Duplicate metadata identity: {row['id']}")
                models[key] = row
                seen.add(key)
        return seen

    candidate_ids = consume_rows(cursor)
    print(f"Selected {len(candidate_ids):,} public candidates and tracked parents", flush=True)
    unavailable: set[str] = set()
    frontier = set(candidate_ids)
    ancestor_rounds = []
    for depth in range(max_ancestor_depth):
        missing = set().union(*(parent_keys(models[key]) for key in frontier)) - models.keys() - unavailable
        if not missing:
            break
        connection.execute("CREATE OR REPLACE TABLE missing_ids AS SELECT CAST(unnest(?) AS VARCHAR) AS id", [sorted(missing)])
        cursor = connection.execute(
            f"SELECT {columns} FROM read_parquet(?) WHERE private IS NOT TRUE "
            "AND lower(id) IN (SELECT id FROM missing_ids)", [str(input_parquet)]
        )
        frontier = consume_rows(cursor)
        unavailable.update(missing - frontier)
        ancestor_rounds.append({"depth": depth + 1, "requested": len(missing), "found": len(frontier)})
        print(f"Ancestor depth {depth + 1}: {len(frontier):,} public rows recovered", flush=True)
    else:
        remaining = set().union(*(parent_keys(models[key]) for key in frontier)) - models.keys() - unavailable
        if remaining:
            raise ValueError("Ancestor closure exceeded max_ancestor_depth; raise the explicit limit")

    if metrics_parquet is not None:
        connection.execute("CREATE OR REPLACE TABLE selected_ids AS SELECT CAST(unnest(?) AS VARCHAR) AS id", [list(models)])
        metrics = connection.execute(
            "SELECT id,downloads,downloads_all_time,sha FROM read_parquet(?) "
            "WHERE lower(id) IN (SELECT id FROM selected_ids)", [str(metrics_parquet)]
        )
        seen_metrics = set()
        while batch := metrics.fetchmany(5000):
            for model_id, downloads, downloads_all_time, revision in batch:
                key = model_id.casefold()
                if key in seen_metrics:
                    raise ValueError(f"Duplicate metrics identity: {model_id}")
                seen_metrics.add(key)
                models[key].update({
                    "downloads": downloads, "downloads_all_time": downloads_all_time,
                    "sha": revision, "metrics_date": metrics_date,
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for key in sorted(models):
            line = (json.dumps(models[key], ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
            stream.write(line)
            digest.update(line)
    summary = {
        "schema_version": 1,
        "dataset": "interconnects/hf-dumps",
        "dataset_revision": dataset_revision,
        "input_sha256": sha256(input_parquet),
        "input_bytes": input_parquet.stat().st_size,
        "primary_sha256": sha256(primary), "extra_sha256": sha256(extra),
        "input_rows": total, "public_input_rows": public,
        "initial_candidates_and_parents": len(candidate_ids),
        "ancestor_rows": len(models) - len(candidate_ids),
        "ancestor_rounds": ancestor_rounds,
        "unavailable_or_private_ancestor_ids": len(unavailable),
        "rows": len(models), "output_sha256": digest.hexdigest(),
        "output_bytes": temporary.stat().st_size,
        "selection": {
            "aliases": list(ALIASES), "artifact_pattern": ARTIFACT_PATTERN,
            "name_pattern": NAME_PATTERN, "tracked_parent_name_count": len(parent_names),
            "quantization_config_fields": ["config_json", "card_data_json"],
            "private_rows": "excluded", "disabled_rows": "retained for catalog rejection",
            "name_matches": "review candidates only; never proof of format or parentage",
            "ancestors": "all declared metadata/tag/card bases, with no inferred name links",
            "card_fields": sorted(CARD_FIELDS),
        },
        "metrics": ({"date": metrics_date, "sha256": sha256(metrics_parquet),
                     "purpose": "review priority only; no download eligibility cutoff"}
                    if metrics_parquet else None),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = manifest.with_name(manifest.name + ".tmp")
    manifest_temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    manifest_temporary.replace(manifest)
    connection.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True, type=Path)
    parser.add_argument("--primary", default=Path("models.csv"), type=Path)
    parser.add_argument("--extra", default=Path("extra_models.csv"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--metrics-parquet", type=Path)
    parser.add_argument("--metrics-date")
    parser.add_argument("--max-ancestor-depth", type=int, default=16)
    arguments = vars(parser.parse_args())
    summary = extract_snapshot(**arguments)
    print(f"Wrote {summary['rows']:,} rows ({summary['output_bytes']:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
