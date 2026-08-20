#!/usr/bin/env python3
"""Validate canonical Hugging Face release-date corrections."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


METADATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = METADATA_DIR / "release_date_corrections.csv"
HEADERS = ("model_id", "hf_created_at", "actual_release_date", "notes")
MODEL_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ReleaseDateValidationError(ValueError):
    """Raised when canonical release-date metadata is invalid."""


@dataclass(frozen=True)
class ReleaseDateCorrectionRow:
    model_id: str
    hf_created_at: date
    actual_release_date: date
    notes: str


def _parse_iso_date(value: str, *, field: str, file_path: Path, line_number: int) -> date:
    if value != value.strip() or not ISO_DATE_RE.fullmatch(value):
        raise ReleaseDateValidationError(
            f"{file_path}:{line_number}: {field} must use YYYY-MM-DD"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseDateValidationError(
            f"{file_path}:{line_number}: {field} is not a valid date"
        ) from exc
    if parsed.isoformat() != value:
        raise ReleaseDateValidationError(
            f"{file_path}:{line_number}: {field} must use YYYY-MM-DD"
        )
    return parsed


def load_and_validate(csv_path: Path = DEFAULT_CSV_PATH) -> list[ReleaseDateCorrectionRow]:
    """Read the registry and enforce its public metadata contract."""
    try:
        handle = csv_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ReleaseDateValidationError(f"cannot read {csv_path}: {exc}") from exc

    with handle:
        reader = csv.reader(handle)
        header_found = False
        rows: list[ReleaseDateCorrectionRow] = []
        seen: set[str] = set()

        for line_number, fields in enumerate(reader, start=1):
            if not fields or fields[0].startswith("#"):
                continue
            if not header_found:
                if tuple(fields) != HEADERS:
                    actual = ",".join(fields) or "<missing>"
                    raise ReleaseDateValidationError(
                        f"unexpected CSV header: {actual}; expected {','.join(HEADERS)}"
                    )
                header_found = True
                continue
            if len(fields) != len(HEADERS):
                raise ReleaseDateValidationError(
                    f"{csv_path}:{line_number}: row has an unexpected number of columns"
                )

            model_id, created_raw, released_raw, notes = fields
            if model_id != model_id.strip() or not MODEL_ID_RE.fullmatch(model_id):
                raise ReleaseDateValidationError(
                    f"{csv_path}:{line_number}: model_id must be an exact org/checkpoint ID"
                )
            if model_id in seen:
                raise ReleaseDateValidationError(
                    f"{csv_path}:{line_number}: duplicate model_id {model_id!r}"
                )
            seen.add(model_id)

            hf_created_at = _parse_iso_date(
                created_raw,
                field="hf_created_at",
                file_path=csv_path,
                line_number=line_number,
            )
            actual_release_date = _parse_iso_date(
                released_raw,
                field="actual_release_date",
                file_path=csv_path,
                line_number=line_number,
            )
            if hf_created_at == actual_release_date:
                raise ReleaseDateValidationError(
                    f"{csv_path}:{line_number}: correction dates must differ"
                )
            if notes != notes.strip():
                raise ReleaseDateValidationError(
                    f"{csv_path}:{line_number}: notes must not contain outer whitespace"
                )
            if not notes:
                raise ReleaseDateValidationError(
                    f"{csv_path}:{line_number}: notes must explain the correction"
                )

            rows.append(
                ReleaseDateCorrectionRow(
                    model_id=model_id,
                    hf_created_at=hf_created_at,
                    actual_release_date=actual_release_date,
                    notes=notes,
                )
            )

    if not header_found:
        raise ReleaseDateValidationError("release_date_corrections.csv is missing its header")
    if not rows:
        raise ReleaseDateValidationError(
            "release_date_corrections.csv must contain at least one row"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args(argv)

    try:
        rows = load_and_validate(args.csv)
    except ReleaseDateValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Validated {len(rows)} release-date correction rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
