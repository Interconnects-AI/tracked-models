#!/usr/bin/env python3
"""Validate the canonical model-parameter metadata in model_parameters.csv."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


METADATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = METADATA_DIR / "model_parameters.csv"
HEADERS = (
    "model_id",
    "total_params_b",
    "active_params_b",
    "is_moe",
    "notes",
)
DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
MODEL_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


class MetadataValidationError(ValueError):
    """Raised when canonical parameter metadata is invalid."""


@dataclass(frozen=True)
class ParameterRow:
    model_id: str
    total_literal: str
    total: Decimal
    active_literal: str
    active: Decimal | None
    is_moe: bool | None
    notes: str


def _parse_positive_decimal(
    value: str,
    *,
    field: str,
    line_number: int,
    required: bool,
) -> Decimal | None:
    if not value:
        if required:
            raise MetadataValidationError(
                f"line {line_number}: {field} must not be blank"
            )
        return None
    if value != value.strip() or not DECIMAL_RE.fullmatch(value):
        raise MetadataValidationError(
            f"line {line_number}: {field} must be a plain positive decimal"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise MetadataValidationError(
            f"line {line_number}: {field} is not a valid decimal"
        ) from exc
    if not parsed.is_finite() or parsed <= 0:
        raise MetadataValidationError(
            f"line {line_number}: {field} must be greater than zero"
        )
    return parsed


def _parse_is_moe(value: str, *, line_number: int) -> bool | None:
    if value == "":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise MetadataValidationError(
        f"line {line_number}: is_moe must be true, false, or blank"
    )


def load_and_validate(csv_path: Path) -> list[ParameterRow]:
    """Read the canonical CSV and enforce its public metadata contract."""
    try:
        handle = csv_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise MetadataValidationError(f"cannot read {csv_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != HEADERS:
            actual = ",".join(reader.fieldnames or ()) or "<missing>"
            raise MetadataValidationError(
                f"unexpected CSV header: {actual}; expected {','.join(HEADERS)}"
            )

        rows: list[ParameterRow] = []
        seen: set[str] = set()
        for raw in reader:
            line_number = reader.line_num
            if None in raw or any(raw[field] is None for field in HEADERS):
                raise MetadataValidationError(
                    f"line {line_number}: row has an unexpected number of columns"
                )

            model_id = raw["model_id"]
            if model_id != model_id.strip() or not MODEL_ID_RE.fullmatch(model_id):
                raise MetadataValidationError(
                    f"line {line_number}: model_id must be an exact org/checkpoint ID"
                )
            if model_id in seen:
                raise MetadataValidationError(
                    f"line {line_number}: duplicate model_id {model_id!r}"
                )
            seen.add(model_id)

            total = _parse_positive_decimal(
                raw["total_params_b"],
                field="total_params_b",
                line_number=line_number,
                required=True,
            )
            active = _parse_positive_decimal(
                raw["active_params_b"],
                field="active_params_b",
                line_number=line_number,
                required=False,
            )
            assert total is not None
            if active is not None and active > total:
                raise MetadataValidationError(
                    f"line {line_number}: active_params_b must not exceed total_params_b"
                )

            is_moe = _parse_is_moe(raw["is_moe"], line_number=line_number)

            notes = raw["notes"]
            if notes != notes.strip():
                raise MetadataValidationError(
                    f"line {line_number}: notes must not contain outer whitespace"
                )

            rows.append(
                ParameterRow(
                    model_id=model_id,
                    total_literal=raw["total_params_b"],
                    total=total,
                    active_literal=raw["active_params_b"],
                    active=active,
                    is_moe=is_moe,
                    notes=notes,
                )
            )

    if not rows:
        raise MetadataValidationError("model_parameters.csv must contain at least one row")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args(argv)

    try:
        rows = load_and_validate(args.csv)
    except MetadataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Validated {len(rows)} parameter rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
