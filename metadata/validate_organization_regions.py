#!/usr/bin/env python3
"""Validate canonical Hugging Face organization-region metadata."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "metadata"
DEFAULT_CSV_PATH = METADATA_DIR / "organization_regions.csv"
DEFAULT_MODEL_LIST_PATHS = (
    REPO_ROOT / "models.csv",
    REPO_ROOT / "extra_models.csv",
)
HEADERS = ("hf_org", "region", "notes")
MODEL_LIST_HEADERS = ("org", "model", "modelId")
REGIONS = frozenset({"us", "china", "eu", "other", "unknown"})
HF_ORG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class OrganizationRegionValidationError(ValueError):
    """Raised when canonical organization-region metadata is invalid."""


@dataclass(frozen=True)
class OrganizationRegionRow:
    hf_org: str
    region: str
    notes: str


def _validate_hf_org(value: str, *, file_path: Path, line_number: int) -> None:
    if (
        value != value.strip()
        or not HF_ORG_RE.fullmatch(value)
        or value.endswith(("-", "."))
        or "--" in value
        or ".." in value
    ):
        raise OrganizationRegionValidationError(
            f"{file_path}:{line_number}: hf_org must be an exact Hugging Face namespace"
        )


def _read_tracked_orgs(model_list_paths: Sequence[Path]) -> set[str]:
    tracked_orgs: set[str] = set()

    for file_path in model_list_paths:
        try:
            handle = file_path.open(newline="", encoding="utf-8")
        except OSError as exc:
            raise OrganizationRegionValidationError(
                f"cannot read tracked model list {file_path}: {exc}"
            ) from exc

        with handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MODEL_LIST_HEADERS:
                actual = ",".join(reader.fieldnames or ()) or "<missing>"
                raise OrganizationRegionValidationError(
                    f"unexpected tracked model header in {file_path}: {actual}; "
                    f"expected {','.join(MODEL_LIST_HEADERS)}"
                )

            for raw in reader:
                line_number = reader.line_num
                if None in raw or any(raw[field] is None for field in MODEL_LIST_HEADERS):
                    raise OrganizationRegionValidationError(
                        f"{file_path}:{line_number}: row has an unexpected number of columns"
                    )
                hf_org = raw["org"]
                _validate_hf_org(
                    hf_org,
                    file_path=file_path,
                    line_number=line_number,
                )
                tracked_orgs.add(hf_org)

    if not tracked_orgs:
        raise OrganizationRegionValidationError(
            "tracked model lists must contain at least one organization"
        )
    return tracked_orgs


def load_and_validate(
    csv_path: Path = DEFAULT_CSV_PATH,
    model_list_paths: Sequence[Path] = DEFAULT_MODEL_LIST_PATHS,
) -> list[OrganizationRegionRow]:
    """Read the registry and enforce its public metadata contract."""
    try:
        handle = csv_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise OrganizationRegionValidationError(f"cannot read {csv_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != HEADERS:
            actual = ",".join(reader.fieldnames or ()) or "<missing>"
            raise OrganizationRegionValidationError(
                f"unexpected CSV header: {actual}; expected {','.join(HEADERS)}"
            )

        rows: list[OrganizationRegionRow] = []
        seen_exact: set[str] = set()
        seen_casefolded: dict[str, str] = {}
        for raw in reader:
            line_number = reader.line_num
            if None in raw or any(raw[field] is None for field in HEADERS):
                raise OrganizationRegionValidationError(
                    f"{csv_path}:{line_number}: row has an unexpected number of columns"
                )

            hf_org = raw["hf_org"]
            _validate_hf_org(hf_org, file_path=csv_path, line_number=line_number)
            if hf_org in seen_exact:
                raise OrganizationRegionValidationError(
                    f"{csv_path}:{line_number}: duplicate hf_org {hf_org!r}"
                )
            seen_exact.add(hf_org)

            folded = hf_org.casefold()
            if folded in seen_casefolded:
                original = seen_casefolded[folded]
                raise OrganizationRegionValidationError(
                    f"{csv_path}:{line_number}: hf_org {hf_org!r} conflicts "
                    f"case-insensitively with {original!r}"
                )
            seen_casefolded[folded] = hf_org

            region = raw["region"]
            if region not in REGIONS:
                raise OrganizationRegionValidationError(
                    f"{csv_path}:{line_number}: region must be one of "
                    f"{', '.join(sorted(REGIONS))}"
                )

            notes = raw["notes"]
            if notes != notes.strip():
                raise OrganizationRegionValidationError(
                    f"{csv_path}:{line_number}: notes must not contain outer whitespace"
                )
            if region == "unknown" and not notes:
                raise OrganizationRegionValidationError(
                    f"{csv_path}:{line_number}: unknown region requires explanatory notes"
                )

            rows.append(
                OrganizationRegionRow(
                    hf_org=hf_org,
                    region=region,
                    notes=notes,
                )
            )

    if not rows:
        raise OrganizationRegionValidationError(
            "organization_regions.csv must contain at least one row"
        )

    expected_order = sorted(rows, key=lambda row: (row.hf_org.casefold(), row.hf_org))
    if rows != expected_order:
        raise OrganizationRegionValidationError(
            "organization_regions.csv must be sorted case-insensitively by hf_org"
        )

    tracked_orgs = _read_tracked_orgs(model_list_paths)
    missing = sorted(tracked_orgs - seen_exact, key=lambda value: (value.casefold(), value))
    if missing:
        raise OrganizationRegionValidationError(
            "organization_regions.csv is missing tracked organizations: "
            + ", ".join(missing)
        )

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--model-list",
        action="append",
        dest="model_lists",
        type=Path,
        help="tracked-models-compatible CSV to require coverage for; repeatable",
    )
    args = parser.parse_args(argv)

    model_list_paths = tuple(args.model_lists or DEFAULT_MODEL_LIST_PATHS)
    try:
        rows = load_and_validate(args.csv, model_list_paths)
    except (OrganizationRegionValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tracked_orgs = _read_tracked_orgs(model_list_paths)
    print(
        f"Validated {len(rows)} organization-region rows covering "
        f"{len(tracked_orgs)} tracked organizations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
