#!/usr/bin/env python3
"""Audit ranked public HF metadata against the separate local-model registry.

Input is normalized metadata JSONL plus nonnegative integer ``downloads`` and/or
``downloadsAllTime`` (``downloads_all_time`` also accepted). Use separate runs
for ranking sources with different dates. Missing metrics are counted, never
filled with zero. Input ``rank``/``source_rank`` is preserved separately: computed
ranks describe only the supplied input, whose global coverage must be documented
by its collection manifest. This script performs no network calls or source edits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
import heapq
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

if __package__:
    from . import local_models as core
else:
    import local_models as core

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("downloads", "downloadsAllTime")
MODEL_TASKS = {"text-generation", "image-text-to-text"}
DISPOSITIONS = (
    "canonical_main",
    "catalog_covered_local",
    "unresolved_local_candidate",
    "outside_model_scope",
    "no_local_format_evidence",
    "unavailable",
)


@dataclass
class Ranked:
    value: int
    key: str
    facts: dict = field(compare=False)

    def __lt__(self, other: Ranked) -> bool:
        # The heap root is the worst retained row: lowest metric, then largest ID.
        return (
            self.value < other.value
            if self.value != other.value
            else self.key > other.key
        )


def metric_value(row: dict, metric: str) -> int | None:
    names = (
        (metric, "downloads_all_time") if metric == "downloadsAllTime" else (metric,)
    )
    values = [row[name] for name in names if row.get(name) is not None]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(
            f"{row.get('id', row.get('modelId'))}: invalid {metric} counter"
        )
    if len(set(values)) > 1:
        raise ValueError(
            f"Conflicting downloadsAllTime/downloads_all_time for {row.get('id')}"
        )
    return values[0] if values else None


def compact_facts(raw: dict) -> dict:
    row = core.normalize_metadata(raw)
    if row["private"]:
        raise ValueError(
            "Private metadata cannot be exported in a public popularity audit"
        )
    pipeline = raw.get("pipeline_tag")
    if pipeline is not None and not isinstance(pipeline, str):
        raise ValueError(f"Invalid pipeline_tag for {row['id']}")
    ranks = {
        name: raw[name]
        for name in ("source_rank", "rank", "audit_rank", "lifetime_rank")
        if raw.get(name) is not None
    }
    if any(type(value) is not int or value < 1 for value in ranks.values()):
        raise ValueError(f"Invalid source rank for {row['id']}")
    source_rank = next(iter(ranks.values()), None)
    return {
        "model_id": row["id"],
        "metrics": {metric: metric_value(raw, metric) for metric in METRICS},
        "source_rank": source_rank,
        "source_ranks": {
            metric: raw[field]
            for metric, field in (
                ("downloads", "audit_rank"),
                ("downloadsAllTime", "lifetime_rank"),
            )
            if raw.get(field) is not None
        },
        "pipeline_tag": pipeline,
        "model_task_tags": sorted(
            MODEL_TASKS.intersection(tag.casefold() for tag in row["tags"])
        ),
        "format_evidence": core.format_evidence(row),
        "base_evidence": core.base_evidence(row),
        "name_hint": bool(core.FORMAT_NAME_PATTERN.search(row["id"].casefold())),
        "disabled": row["disabled"],
        "source_url": row["source_url"],
        "sha": row["sha"],
        "source_revision": raw.get("source_revision"),
        "observed_at": row["observed_at"],
    }


def select_ranked(path: Path, metrics: tuple[str, ...], top: int) -> tuple[dict, dict]:
    heaps: dict[str, list[Ranked]] = {metric: [] for metric in metrics}
    missing = Counter()
    seen = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{number}: expected metadata object")
            facts = compact_facts(raw)
            key = facts["model_id"].casefold()
            if key in seen:
                raise ValueError(f"Duplicate ranked repository: {facts['model_id']}")
            seen.add(key)
            for metric, heap in heaps.items():
                value = facts["metrics"][metric]
                if value is None:
                    missing[metric] += 1
                    continue
                ranked = Ranked(value, key, facts)
                if len(heap) < top:
                    heapq.heappush(heap, ranked)
                elif heap[0] < ranked:
                    heapq.heapreplace(heap, ranked)
    selected, rankings = {}, {}
    for metric, heap in heaps.items():
        if not heap:
            raise ValueError(f"No nonmissing values for selected metric {metric}")
        ordered = sorted(heap, key=lambda row: (-row.value, row.key))
        for rank, entry in enumerate(ordered, 1):
            selected.setdefault(entry.key, {**entry.facts, "ranks": {}})["ranks"][
                metric
            ] = rank
        rankings[metric] = {
            "requested_top": top,
            "selected_repositories": len(ordered),
            "input_repositories": len(seen),
            "missing_metric_repositories": missing[metric],
            "rankable_input_repositories": len(seen) - missing[metric],
            "cutoff_downloads": ordered[-1].value,
            "order": "counter descending, case-insensitive repository ID ascending",
        }
    return selected, rankings


def selected_csv(path: Path, id_column: str, wanted: set[str]) -> dict:
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise ValueError(f"Missing {id_column} column in {path}")
        for row in reader:
            key = row[id_column].casefold()
            if key in wanted:
                if key in rows:
                    raise ValueError(
                        f"Duplicate selected repository in {path}: {row[id_column]}"
                    )
                rows[key] = row
    return rows


def catalog_coverage(
    path: Path, wanted: set[str], parent_ids: set[str]
) -> tuple[dict, dict]:
    selected, parent_counts = {}, {key: Counter() for key in parent_ids}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "modelId",
            "parent_model_id",
            "local_formats",
            "variant_category",
            "is_modified",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Catalog CSV is missing required coverage columns")
        for row in reader:
            key, parent = row["modelId"].casefold(), row["parent_model_id"].casefold()
            if key in wanted:
                if key in selected:
                    raise ValueError(
                        f"Duplicate selected catalog repository: {row['modelId']}"
                    )
                selected[key] = {field: row[field] for field in sorted(required)}
            if parent in parent_counts:
                counts = parent_counts[parent]
                counts["local_variant_repos"] += 1
                counts["conversion_repos"] += row["variant_category"] == "conversion"
                counts["modified_repos"] += row["is_modified"] == "true"
    return selected, {
        key: {
            field: counts[field]
            for field in ("local_variant_repos", "conversion_repos", "modified_repos")
        }
        for key, counts in parent_counts.items()
    }


def classify(
    facts: dict, canonical: dict, catalog: dict | None, candidate: dict | None
) -> dict:
    key = facts["model_id"].casefold()
    bases = facts["base_evidence"]["base_model_ids"]
    declared_tracked = [
        canonical[base.casefold()]["modelId"]
        for base in bases
        if base.casefold() in canonical
    ]
    formats = sorted({entry["format"] for entry in facts["format_evidence"]})
    has_local_signal = bool(formats or facts["name_hint"] or candidate)
    pipeline = (facts["pipeline_tag"] or "").casefold()
    outside_task = bool(
        pipeline and pipeline not in MODEL_TASKS and not facts["model_task_tags"]
    )
    if facts["disabled"]:
        disposition, reason = "unavailable", "public_repository_disabled"
    elif key in canonical:
        disposition, reason = "canonical_main", "in_canonical_tracked_lists"
    elif catalog:
        disposition, reason = "catalog_covered_local", "in_local_model_catalog"
    elif outside_task and not declared_tracked:
        disposition, reason = (
            "outside_model_scope",
            "explicit_pipeline_outside_llm_vlm_tasks",
        )
    elif has_local_signal:
        disposition = "unresolved_local_candidate"
        reason = (
            candidate["reason"]
            if candidate
            else "supported_format_not_in_catalog"
            if formats
            else "name_hint_requires_metadata_review"
        )
    else:
        disposition, reason = (
            "no_local_format_evidence",
            "no_supported_format_in_supplied_metadata",
        )
    return {
        **facts,
        "local_formats": formats,
        "disposition": disposition,
        "reason": reason,
        "declared_tracked_parents": declared_tracked,
        "canonical_list": canonical[key]["parent_list"] if key in canonical else None,
        "catalog_entry": catalog,
        "candidate_entry": {
            field: candidate.get(field, "")
            for field in ("reason", "local_formats", "base_model_ids", "lineage")
        }
        if candidate
        else None,
        "review_priority": "supported_format"
        if disposition == "unresolved_local_candidate" and formats
        else "unconfirmed_hint"
        if disposition == "unresolved_local_candidate"
        else None,
    }


def audit(
    metadata: Path,
    catalog: Path,
    models: Path,
    extra: Path,
    *,
    metrics: tuple[str, ...] = METRICS,
    top: int = 1000,
    candidates: Path | None = None,
    source_manifest: Path | None = None,
) -> tuple[list[dict], dict]:
    if (
        top < 1
        or not metrics
        or len(set(metrics)) != len(metrics)
        or set(metrics) - set(METRICS)
    ):
        raise ValueError("Choose a positive cutoff and distinct supported metrics")
    selected, rankings = select_ranked(metadata, metrics, top)
    canonical = {
        row["modelId"].casefold(): row for row in core.load_parents(models, extra)
    }
    wanted = set(selected)
    entries, counts = catalog_coverage(catalog, wanted, wanted.intersection(canonical))
    unresolved = selected_csv(candidates, "model_id", wanted) if candidates else {}
    rows = []
    for key, facts in sorted(selected.items()):
        row = classify(facts, canonical, entries.get(key), unresolved.get(key))
        row["parent_catalog_coverage"] = counts.get(key)
        rows.append(row)
    for metric, ranking in rankings.items():
        cohort = [row for row in rows if metric in row["ranks"]]
        tally = Counter(row["disposition"] for row in cohort)
        ranking["dispositions"] = {label: tally[label] for label in DISPOSITIONS}
        ranking["unresolved_supported_format"] = sum(
            row["review_priority"] == "supported_format" for row in cohort
        )
        ranking["unresolved_name_or_prior_candidate_only"] = sum(
            row["review_priority"] == "unconfirmed_hint" for row in cohort
        )
    paths = {
        "metadata": metadata,
        "catalog": catalog,
        "models": models,
        "extra_models": extra,
        "classification_rules": Path(core.__file__),
        "audit_script": Path(__file__),
    }
    if candidates:
        paths["candidates"] = candidates
    if source_manifest:
        paths["source_manifest"] = source_manifest
    source = (
        json.loads(source_manifest.read_text(encoding="utf-8"))
        if source_manifest
        else None
    )
    inputs = {
        key: {"name": path.name, "sha256": core.file_sha(path)}
        for key, path in paths.items()
    }
    if source is not None and not isinstance(source, dict):
        raise ValueError("Source manifest must be an object")
    if source:
        expected_rows = source.get("rows")
        if (
            expected_rows is not None
            and expected_rows != rankings[metrics[0]]["input_repositories"]
        ):
            raise ValueError("Source manifest row count does not match ranked metadata")
        claimed = source.get("sha256", source.get("output_sha256"))
        if claimed and claimed != inputs["metadata"]["sha256"]:
            raise ValueError("Source manifest hash does not match ranked metadata")
    report = {
        "schema_version": 1,
        "inputs": inputs,
        "source": source,
        "scope": "Ranked supplied public metadata only; global ranking completeness and observation date depend on the collection source. Separate runs are required for different source dates.",
        "classification": "Catalog/canonical membership is case-insensitive. Local format evidence reuses catalog rules. Names are review hints only. Missing format evidence does not establish that a repository has no local artifacts. Outside-scope task labels are metadata observations, not independent model-card review.",
        "download_caveat": "HF download counters are repository request statistics, not unique users; formats can use different counting rules.",
        "rankings": rankings,
        "unique_selected_repositories": len(rows),
    }
    return rows, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "local-models.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "models.csv")
    parser.add_argument("--extra", type=Path, default=ROOT / "extra_models.csv")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--metric", action="append", choices=METRICS)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="Compare artifacts without writing"
    )
    args = parser.parse_args(argv)
    inputs = {
        path.resolve()
        for path in (
            args.metadata,
            args.catalog,
            args.models,
            args.extra,
            args.candidates,
            args.source_manifest,
        )
        if path
    }
    outputs = {args.output.resolve(), args.report.resolve()}
    if len(outputs) != 2 or inputs.intersection(outputs):
        parser.error(
            "Output and report must be distinct from each other and every input"
        )
    rows, report = audit(
        args.metadata,
        args.catalog,
        args.models,
        args.extra,
        metrics=tuple(args.metric or METRICS),
        top=args.top,
        candidates=args.candidates,
        source_manifest=args.source_manifest,
    )
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for row in rows
    ).encode()
    report["output_sha256"] = hashlib.sha256(payload).hexdigest()
    expected = (
        (args.output, payload),
        (
            args.report,
            (
                json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
            ).encode(),
        ),
    )
    if args.check:
        for path, content in expected:
            if not path.exists() or path.read_bytes() != content:
                raise ValueError(f"Popularity audit differs: {path}")
        print(f"Verified {len(rows):,} ranked repository dispositions")
        return
    for path, content in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    print(f"Audited {len(rows):,} ranked repositories")


if __name__ == "__main__":
    main()
