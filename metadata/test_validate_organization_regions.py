"""Tests for the canonical organization-region metadata contract."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from metadata.validate_organization_regions import (
    HEADERS,
    MODEL_LIST_HEADERS,
    OrganizationRegionValidationError,
    load_and_validate,
)


class OrganizationRegionMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.csv_path = root / "organization_regions.csv"
        self.models_path = root / "models.csv"
        self.extra_models_path = root / "extra_models.csv"
        self.write_model_lists(["Alpha"])

    @property
    def model_list_paths(self) -> tuple[Path, Path]:
        return (self.models_path, self.extra_models_path)

    def write_registry(
        self,
        rows: list[tuple[str, str, str]],
        headers: tuple[str, ...] = HEADERS,
    ) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(rows)

    def write_model_lists(self, orgs: list[str]) -> None:
        for index, path in enumerate(self.model_list_paths):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(MODEL_LIST_HEADERS)
                if index == 0:
                    for number, org in enumerate(orgs, start=1):
                        writer.writerow((org, f"Model-{number}", f"{org}/Model-{number}"))

    def validate(self):
        return load_and_validate(self.csv_path, self.model_list_paths)

    def test_repository_registry_is_valid_and_preserves_reviewed_unknowns(self) -> None:
        rows = load_and_validate()
        regions = {row.hf_org: row.region for row in rows}

        self.assertGreaterEqual(len(rows), 118)
        self.assertEqual(regions["AIDC-AI"], "china")
        self.assertEqual(regions["THUDM"], "china")
        self.assertEqual(regions["mistral-community"], "eu")
        self.assertEqual(regions["docling-project"], "eu")
        for hf_org in ("allura-forge", "bigcode", "EleutherAI", "open-thoughts"):
            self.assertEqual(regions[hf_org], "unknown")

    def test_valid_registry_allows_extra_orgs_and_quoted_notes(self) -> None:
        self.write_registry(
            [
                ("Alpha", "us", ""),
                ("beta", "unknown", "Distributed, international collaboration."),
                ("Extra", "eu", ""),
            ]
        )
        self.write_model_lists(["Alpha", "beta"])

        rows = self.validate()

        self.assertEqual([row.hf_org for row in rows], ["Alpha", "beta", "Extra"])
        self.assertEqual(rows[1].notes, "Distributed, international collaboration.")

    def test_header_is_exact(self) -> None:
        self.write_registry(
            [("Alpha", "us", "")],
            headers=("organization", "region", "notes"),
        )

        with self.assertRaisesRegex(OrganizationRegionValidationError, "unexpected CSV header"):
            self.validate()

    def test_exact_and_casefolded_duplicates_are_rejected(self) -> None:
        self.write_registry([("Alpha", "us", ""), ("Alpha", "eu", "")])
        with self.assertRaisesRegex(OrganizationRegionValidationError, "duplicate hf_org"):
            self.validate()

        self.write_registry([("Alpha", "us", ""), ("alpha", "eu", "")])
        with self.assertRaisesRegex(OrganizationRegionValidationError, "case-insensitively"):
            self.validate()

    def test_hf_org_must_be_an_exact_namespace(self) -> None:
        invalid_values = (
            " org",
            "org/name",
            "org--name",
            "org..name",
            "org-",
            "org.",
            "_org",
            "a" * 97,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                self.write_registry([(value, "us", "")])
                with self.assertRaisesRegex(
                    OrganizationRegionValidationError,
                    "exact Hugging Face namespace",
                ):
                    self.validate()

    def test_region_uses_the_fixed_enum(self) -> None:
        self.write_registry([("Alpha", "USA", "")])

        with self.assertRaisesRegex(OrganizationRegionValidationError, "region must be one of"):
            self.validate()

    def test_unknown_requires_notes_and_all_notes_are_trimmed(self) -> None:
        self.write_registry([("Alpha", "unknown", "")])
        with self.assertRaisesRegex(OrganizationRegionValidationError, "requires explanatory notes"):
            self.validate()

        self.write_registry([("Alpha", "us", " outer whitespace")])
        with self.assertRaisesRegex(OrganizationRegionValidationError, "outer whitespace"):
            self.validate()

    def test_rows_must_be_sorted_case_insensitively(self) -> None:
        self.write_registry([("beta", "us", ""), ("Alpha", "us", "")])
        self.write_model_lists(["Alpha", "beta"])

        with self.assertRaisesRegex(OrganizationRegionValidationError, "sorted case-insensitively"):
            self.validate()

    def test_every_tracked_org_requires_an_exact_registry_row(self) -> None:
        self.write_registry([("Alpha", "us", "")])
        self.write_model_lists(["Alpha", "beta"])

        with self.assertRaisesRegex(
            OrganizationRegionValidationError,
            "missing tracked organizations: beta",
        ):
            self.validate()

    def test_tracked_model_headers_are_validated(self) -> None:
        self.write_registry([("Alpha", "us", "")])
        self.models_path.write_text("org,model\nAlpha,Model\n", encoding="utf-8")

        with self.assertRaisesRegex(
            OrganizationRegionValidationError,
            "unexpected tracked model header",
        ):
            self.validate()


if __name__ == "__main__":
    unittest.main()
