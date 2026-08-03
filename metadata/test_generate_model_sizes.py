"""Tests for the canonical model-parameter metadata contract."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from metadata.generate_model_sizes import (
    BEGIN_MARKER,
    END_MARKER,
    HEADERS,
    LEGACY_NOTE_PREFIX,
    MetadataValidationError,
    load_and_validate,
    render_manual_sizes,
    replace_generated_block,
)


class ModelParameterMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.csv_path = Path(self.temp_dir.name) / "model_parameters.csv"
        self.valid_row = {
            "model_id": "org/checkpoint",
            "total_params_b": "10.5",
            "active_params_b": "3",
            "is_moe": "true",
            "count_status": "verified",
            "notes": "Official model card value.",
        }

    def write_rows(self, rows: list[dict[str, str]], headers=HEADERS) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def assert_invalid(self, **changes: str) -> None:
        row = {**self.valid_row, **changes}
        self.write_rows([row])
        with self.assertRaises(MetadataValidationError):
            load_and_validate(self.csv_path)

    def test_valid_rows_and_optional_active_value(self) -> None:
        second = {
            **self.valid_row,
            "model_id": "org/unreviewed-checkpoint",
            "active_params_b": "",
            "is_moe": "",
            "count_status": "needs_source",
        }
        dense = {
            **self.valid_row,
            "model_id": "org/dense-checkpoint",
            "active_params_b": "",
            "is_moe": "false",
        }
        self.write_rows([self.valid_row, second, dense])

        rows = load_and_validate(self.csv_path)

        self.assertEqual(len(rows), 3)
        self.assertEqual(str(rows[0].total), "10.5")
        self.assertIs(rows[0].is_moe, True)
        self.assertIsNone(rows[1].active)
        self.assertIsNone(rows[1].is_moe)
        self.assertIsNone(rows[2].active)
        self.assertIs(rows[2].is_moe, False)

    def test_header_is_exact(self) -> None:
        wrong_headers = tuple(header for header in HEADERS if header != "notes")
        row = {key: value for key, value in self.valid_row.items() if key in wrong_headers}
        self.write_rows([row], headers=wrong_headers)

        with self.assertRaisesRegex(MetadataValidationError, "unexpected CSV header"):
            load_and_validate(self.csv_path)

    def test_duplicate_model_ids_are_rejected(self) -> None:
        self.write_rows([self.valid_row, self.valid_row])

        with self.assertRaisesRegex(MetadataValidationError, "duplicate model_id"):
            load_and_validate(self.csv_path)

    def test_model_id_must_be_exact_checkpoint_id(self) -> None:
        for value in ("checkpoint", "org/model/extra", " org/model", "org/model "):
            with self.subTest(value=value):
                self.assert_invalid(model_id=value)

    def test_parameter_values_are_plain_positive_decimals(self) -> None:
        for value in ("", "0", "-1", ".5", "1e3", " 1"):
            with self.subTest(value=value):
                self.assert_invalid(total_params_b=value)

    def test_active_parameters_must_be_positive_and_not_exceed_total(self) -> None:
        for value in ("0", "-1", "10.6"):
            with self.subTest(value=value):
                self.assert_invalid(active_params_b=value)

    def test_status_is_restricted(self) -> None:
        self.assert_invalid(count_status="inferred")

    def test_is_moe_is_tri_state_and_required_for_publishable_rows(self) -> None:
        self.assert_invalid(is_moe="yes")
        self.assert_invalid(is_moe="")

        for value, expected in (("true", True), ("false", False)):
            with self.subTest(value=value):
                row = {**self.valid_row, "is_moe": value}
                self.write_rows([row])
                self.assertIs(load_and_validate(self.csv_path)[0].is_moe, expected)

        row = {
            **self.valid_row,
            "is_moe": "",
            "count_status": "needs_source",
        }
        self.write_rows([row])
        self.assertIsNone(load_and_validate(self.csv_path)[0].is_moe)

    def test_all_contract_statuses_are_accepted(self) -> None:
        for status in ("verified", "estimated", "needs_source"):
            with self.subTest(status=status):
                row = {**self.valid_row, "count_status": status}
                self.write_rows([row])
                self.assertEqual(load_and_validate(self.csv_path)[0].count_status, status)

    def test_generated_map_uses_only_marked_legacy_rows(self) -> None:
        legacy = {
            **self.valid_row,
            "notes": f"{LEGACY_NOTE_PREFIX} Migrated value.",
        }
        canonical_only = {
            **self.valid_row,
            "model_id": "org/new-checkpoint",
        }
        self.write_rows([legacy, canonical_only])

        rendered = render_manual_sizes(load_and_validate(self.csv_path))

        self.assertIn("'org/checkpoint': 10.5", rendered)
        self.assertNotIn("org/new-checkpoint", rendered)

    def test_generated_markers_must_be_unique(self) -> None:
        module = f"before\n{BEGIN_MARKER}\nold\n{END_MARKER}\nafter\n"
        generated = f"{BEGIN_MARKER}\nnew\n{END_MARKER}"
        self.assertIn("\nnew\n", replace_generated_block(module, generated))

        with self.assertRaises(MetadataValidationError):
            replace_generated_block("no markers", generated)


if __name__ == "__main__":
    unittest.main()
