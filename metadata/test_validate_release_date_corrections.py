"""Tests for the canonical release-date correction metadata contract."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from metadata.validate_release_date_corrections import (
    HEADERS,
    ReleaseDateValidationError,
    load_and_validate,
)


class ReleaseDateCorrectionMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.csv_path = Path(self.temp_dir.name) / "release_date_corrections.csv"
        self.valid_row = (
            "org/checkpoint",
            "2026-01-01",
            "2026-01-02",
            "official launch date",
        )

    def write_rows(
        self,
        rows: list[tuple[str, str, str, str]],
        headers: tuple[str, ...] = HEADERS,
        *,
        comments: bool = False,
    ) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            if comments:
                handle.write("# Manual release date corrections\n")
                handle.write("# Format: model_id,hf_created_at,actual_release_date,notes\n")
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(rows)

    def assert_invalid(self, row: tuple[str, str, str, str], message: str) -> None:
        self.write_rows([row])
        with self.assertRaisesRegex(ReleaseDateValidationError, message):
            load_and_validate(self.csv_path)

    def test_repository_registry_is_valid_and_covers_release_siblings(self) -> None:
        rows = load_and_validate()
        by_id = {row.model_id: row for row in rows}

        expected_qwen = {
            "Qwen/Qwen3.8-2.4T-A95B": date(2026, 8, 8),
            "Qwen/Qwen3.8-2.4T-A95B-FP8": date(2026, 8, 8),
            "Qwen/Qwen3.8-27B": date(2026, 8, 5),
            "Qwen/Qwen3.8-27B-FP8": date(2026, 8, 13),
        }
        expected_ministral = {
            "mistralai/Ministral-3-14B-Base-2512",
            "mistralai/Ministral-3-14B-Instruct-2512",
            "mistralai/Ministral-3-14B-Instruct-2512-BF16",
            "mistralai/Ministral-3-14B-Reasoning-2512",
            "mistralai/Ministral-3-3B-Base-2512",
            "mistralai/Ministral-3-3B-Instruct-2512",
            "mistralai/Ministral-3-3B-Instruct-2512-BF16",
            "mistralai/Ministral-3-3B-Instruct-2512-ONNX",
            "mistralai/Ministral-3-3B-Reasoning-2512",
            "mistralai/Ministral-3-8B-Base-2512",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "mistralai/Ministral-3-8B-Reasoning-2512",
        }

        for model_id, created_at in expected_qwen.items():
            self.assertEqual(by_id[model_id].hf_created_at, created_at)
            expected_release = date(2026, 8, 12 if "2.4T" in model_id else 14)
            self.assertEqual(by_id[model_id].actual_release_date, expected_release)
        for model_id in expected_ministral:
            expected_created = (
                date(2025, 11, 24)
                if model_id.endswith("-ONNX")
                else date(2025, 10, 31)
            )
            self.assertEqual(by_id[model_id].hf_created_at, expected_created)
            self.assertEqual(by_id[model_id].actual_release_date, date(2025, 12, 2))

    def test_comments_and_quoted_notes_are_supported(self) -> None:
        row = (*self.valid_row[:3], "official launch, model card")
        self.write_rows([row], comments=True)

        rows = load_and_validate(self.csv_path)

        self.assertEqual(rows[0].notes, "official launch, model card")

    def test_header_is_exact(self) -> None:
        self.write_rows([self.valid_row], headers=("model", *HEADERS[1:]))
        with self.assertRaisesRegex(ReleaseDateValidationError, "unexpected CSV header"):
            load_and_validate(self.csv_path)

    def test_duplicate_model_ids_are_rejected(self) -> None:
        self.write_rows([self.valid_row, self.valid_row])
        with self.assertRaisesRegex(ReleaseDateValidationError, "duplicate model_id"):
            load_and_validate(self.csv_path)

    def test_model_id_must_be_an_exact_checkpoint_id(self) -> None:
        for value in ("checkpoint", "org/model/extra", " org/model", "org/model "):
            with self.subTest(value=value):
                self.assert_invalid((value, *self.valid_row[1:]), "exact org/checkpoint ID")

    def test_dates_are_strict_valid_iso_dates(self) -> None:
        invalid = ("2026-1-01", "2026-02-30", " 2026-01-01", "not-a-date")
        for value in invalid:
            with self.subTest(field="hf_created_at", value=value):
                self.assert_invalid(
                    (self.valid_row[0], value, *self.valid_row[2:]),
                    "hf_created_at",
                )
            with self.subTest(field="actual_release_date", value=value):
                self.assert_invalid(
                    (*self.valid_row[:2], value, self.valid_row[3]),
                    "actual_release_date",
                )

    def test_correction_dates_must_differ(self) -> None:
        self.assert_invalid(
            (self.valid_row[0], "2026-01-02", "2026-01-02", self.valid_row[3]),
            "correction dates must differ",
        )

    def test_notes_are_required_and_trimmed(self) -> None:
        for value in ("", " leading", "trailing "):
            with self.subTest(value=value):
                self.assert_invalid((*self.valid_row[:3], value), "notes")


if __name__ == "__main__":
    unittest.main()
