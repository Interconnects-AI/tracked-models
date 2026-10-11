#!/usr/bin/env python3
"""Report primary model-size coverage and reject newly unclassified checkpoints."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from decimal import Decimal
from pathlib import Path

from metadata.generate_model_sizes import HEADERS, MODEL_ID_RE, load_and_validate

ROOT = Path(__file__).resolve().parents[1]
MODEL_HEADERS = ("org", "model", "modelId")


def tracked_ids(text: str) -> set[str]:
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != MODEL_HEADERS:
        raise ValueError("Unexpected models.csv header")
    result = set()
    for row in reader:
        model_id = row["modelId"]
        if (
            set(row) != set(MODEL_HEADERS)
            or not model_id
            or not MODEL_ID_RE.fullmatch(model_id)
            or model_id != f"{row['org']}/{row['model']}"
            or model_id != model_id.strip()
            or model_id in result
        ):
            raise ValueError(f"Invalid or duplicate tracked checkpoint: {model_id!r}")
        result.add(model_id)
    return result


def previous_missing(repo: Path, ref: str) -> set[str]:
    def read(path: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", "--end-of-options", f"{ref}:{path}"],
            text=True,
        )

    primary = tracked_ids(read("models.csv"))
    reader = csv.DictReader(io.StringIO(read("metadata/model_parameters.csv")))
    if tuple(reader.fieldnames or ()) != HEADERS:
        raise ValueError("Unexpected historical parameter metadata header")
    return primary - {row["model_id"] for row in reader}


def coverage(primary: set[str], parameters: dict[str, Decimal | float], baseline: set[str] | None = None) -> dict:
    missing = primary - parameters.keys()
    known = primary & parameters.keys()
    return {
        "primary_models": len(primary),
        "known_sizes": len(known),
        "over_100b": sorted(mid for mid in known if parameters[mid] > 100),
        "at_100b_boundary": sorted(mid for mid in known if parameters[mid] == 100),
        "at_or_below_100b": len(known) - sum(parameters[mid] > 100 for mid in known),
        "missing_sizes": sorted(missing),
        "newly_missing_sizes": sorted(missing if baseline is None else missing - baseline),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models.csv")
    parser.add_argument("--parameters", type=Path, default=ROOT / "metadata/model_parameters.csv")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base-ref", help="Compare missing IDs with this Git commit; existing gaps remain reported")
    parser.add_argument("--check", action="store_true", help="Fail on missing sizes, or newly missing sizes with --base-ref")
    parser.add_argument("--json-output", type=Path, help="Write the full missing-ID and >100B lists for review")
    parser.add_argument("--markdown", action="store_true", help="Emit a concise PR/CI summary")
    args = parser.parse_args(argv)
    rows = load_and_validate(args.parameters)
    report = coverage(
        tracked_ids(args.models.read_text()),
        {row.model_id: row.total for row in rows},
        previous_missing(args.repo, args.base_ref) if args.base_ref else None,
    )
    if args.json_output:
        if args.json_output.resolve() in {args.models.resolve(), args.parameters.resolve()}:
            parser.error("Report output must not overwrite an input")
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        print("\n### Primary model parameter coverage\n")
    print(f"Known sizes: {report['known_sizes']}/{report['primary_models']}; "
          f">100B: {len(report['over_100b'])}; missing: {len(report['missing_sizes'])}.")
    if report["at_100b_boundary"]:
        print("Published sizes at exactly 100B (review rounding/component scope): "
              + ", ".join(report["at_100b_boundary"]) + ".")
    new = report["newly_missing_sizes"]
    if args.base_ref:
        print(f"Newly missing sizes relative to {args.base_ref}: {len(new)}.")
    if new:
        print("\nSize review required for:")
        for model_id in new:
            print(f"- {model_id}")
    elif report["missing_sizes"]:
        print("Existing gaps remain; use --json-output for the full review list.")
    return int(args.check and bool(new))


if __name__ == "__main__":
    raise SystemExit(main())
