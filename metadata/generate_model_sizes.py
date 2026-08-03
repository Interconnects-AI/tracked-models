#!/usr/bin/env python3
"""Validate model parameter metadata and generate the compatibility size map."""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse


METADATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = METADATA_DIR / "model_parameters.csv"
DEFAULT_MODULE_PATH = METADATA_DIR / "model_sizes.py"
HEADERS = (
    "model_id",
    "total_params_b",
    "active_params_b",
    "count_status",
    "source_url",
    "notes",
)
ALLOWED_STATUSES = {"verified", "estimated", "needs_source"}
PUBLISHABLE_STATUSES = {"verified", "estimated"}
DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
MODEL_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
LEGACY_NOTE_PREFIX = "Legacy MANUAL_SIZES entry."
BEGIN_MARKER = "# BEGIN GENERATED MANUAL_SIZES"
END_MARKER = "# END GENERATED MANUAL_SIZES"


class MetadataValidationError(ValueError):
    """Raised when canonical parameter metadata is invalid."""


@dataclass(frozen=True)
class ParameterRow:
    model_id: str
    total_literal: str
    total: Decimal
    active_literal: str
    active: Decimal | None
    count_status: str
    source_url: str
    notes: str

    @property
    def is_manual_sizes_compat(self) -> bool:
        return self.notes.startswith(LEGACY_NOTE_PREFIX)


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

            status = raw["count_status"]
            if status not in ALLOWED_STATUSES:
                allowed = ", ".join(sorted(ALLOWED_STATUSES))
                raise MetadataValidationError(
                    f"line {line_number}: count_status must be one of {allowed}"
                )

            source_url = raw["source_url"]
            if source_url != source_url.strip():
                raise MetadataValidationError(
                    f"line {line_number}: source_url must not contain outer whitespace"
                )
            if status in PUBLISHABLE_STATUSES and not source_url:
                raise MetadataValidationError(
                    f"line {line_number}: {status} rows require source_url"
                )
            if source_url:
                parsed_url = urlparse(source_url)
                if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                    raise MetadataValidationError(
                        f"line {line_number}: source_url must be an http(s) URL"
                    )

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
                    count_status=status,
                    source_url=source_url,
                    notes=notes,
                )
            )

    if not rows:
        raise MetadataValidationError("model_parameters.csv must contain at least one row")
    return rows


def render_manual_sizes(rows: list[ParameterRow]) -> str:
    """Render the literal legacy override map from marked canonical CSV rows."""
    compat_rows = [row for row in rows if row.is_manual_sizes_compat]
    if not compat_rows:
        raise MetadataValidationError(
            f"no rows carry the compatibility marker {LEGACY_NOTE_PREFIX!r}"
        )

    lines = [
        BEGIN_MARKER,
        "# Generated by metadata/generate_model_sizes.py from model_parameters.csv.",
        "# Do not edit this block directly.",
        "MANUAL_SIZES = {",
    ]
    lines.extend(
        f"    {row.model_id!r}: {row.total_literal}," for row in compat_rows
    )
    lines.extend(["}", END_MARKER])
    return "\n".join(lines)


def replace_generated_block(module_text: str, generated: str) -> str:
    """Replace the marked block, rejecting missing or duplicate markers."""
    if module_text.count(BEGIN_MARKER) != 1 or module_text.count(END_MARKER) != 1:
        raise MetadataValidationError(
            "model_sizes.py must contain exactly one generated marker pair"
        )
    start = module_text.index(BEGIN_MARKER)
    end = module_text.index(END_MARKER, start) + len(END_MARKER)
    if start >= end:
        raise MetadataValidationError("generated markers are out of order")
    return module_text[:start] + generated + module_text[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate the CSV and fail if model_sizes.py is out of date",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="validate the CSV and regenerate the literal compatibility map",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE_PATH)
    args = parser.parse_args(argv)

    try:
        rows = load_and_validate(args.csv)
        current = args.module.read_text(encoding="utf-8")
        expected_block = render_manual_sizes(rows)
        expected = replace_generated_block(current, expected_block)
    except (MetadataValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    compat_count = sum(row.is_manual_sizes_compat for row in rows)
    if args.check:
        if current != expected:
            diff = difflib.unified_diff(
                current.splitlines(),
                expected.splitlines(),
                fromfile=str(args.module),
                tofile=f"{args.module} (generated)",
                lineterm="",
            )
            print("\n".join(diff), file=sys.stderr)
            print(
                "error: generated MANUAL_SIZES is stale; run with --write",
                file=sys.stderr,
            )
            return 1
        print(
            f"Validated {len(rows)} parameter rows and {compat_count} "
            "MANUAL_SIZES entries."
        )
        return 0

    with args.module.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(expected)
    print(f"Updated {args.module} with {compat_count} MANUAL_SIZES entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
