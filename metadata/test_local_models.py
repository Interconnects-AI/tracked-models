"""Tests for the conversion-only local model catalog."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metadata.local_models import (
    CatalogError,
    build_rows,
    detect_formats,
    rejection,
    validate_catalog,
    write_catalog,
)

PARENTS = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-14B"]
REGIONS = {"qwen": "china", "meta-llama": "us"}
OVERRIDES = {
    "aliases": {"meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"},
    "include": {"someone/odd-name-awq": {"parent": "Qwen/Qwen3-14B", "formats": ["awq"]}},
    "exclude": {"someone/Qwen3-8B-GGUF": "pruned: reviewed."},
}


def child(model_id, parent, *, downloads=500_000, tags=(), relation="quantized", **extra):
    return {
        "id": model_id,
        "tags": [f"base_model:{parent}", f"base_model:{relation}:{parent}", *tags],
        "library_name": extra.pop("library_name", None),
        "downloads_all_time": downloads,
        "created_at": "2025-04-28T12:00:00.000Z",
        "card_data": extra.pop("card_data", {}),
        **extra,
    }


def rows_for(*candidates):
    return build_rows(candidates, PARENTS, set(), REGIONS, OVERRIDES)


class LocalModelTests(unittest.TestCase):
    def test_direct_conversion_credits_the_original_parent(self) -> None:
        [row] = rows_for(("Qwen/Qwen3-8B", child("unsloth/Qwen3-8B-GGUF", "Qwen/Qwen3-8B", tags=["gguf"])))
        self.assertEqual(
            row,
            {
                "org": "unsloth",
                "model": "Qwen3-8B-GGUF",
                "modelId": "unsloth/Qwen3-8B-GGUF",
                "parent_model_id": "Qwen/Qwen3-8B",
                "parent_org": "Qwen",
                "parent_region": "china",
                "local_formats": "gguf",
                "created_at": "2025-04-28",
            },
        )

    def test_formats_come_from_library_tags_config_and_files(self) -> None:
        model = {
            "library_name": "mlx",
            "tags": ["compressed-tensors"],
            "config": {"quantization_config": {"quant_method": "awq"}},
            "siblings": ["model-Q4_K_M.gguf", "README.md"],
        }
        self.assertEqual(detect_formats(model), ["gguf", "mlx", "awq", "compressed-tensors"])

    def test_format_named_only_in_the_repo_name_is_not_evidence(self) -> None:
        model = child("someone/Qwen3-8B-GGUF-soon", "Qwen/Qwen3-8B")
        self.assertEqual(rejection(model, "Qwen/Qwen3-8B", {}), "no_format_evidence")

    def test_modified_descendants_are_excluded(self) -> None:
        cases = {
            "someone/Qwen3-8B-REAP-GGUF": {},
            "someone/Qwen3-8B-abliterated-GGUF": {},
            "someone/Qwen3-8B-GGUF-x": {"relation": "finetune"},
            "someone/Qwen3-8B-mlx": {"tags": ["gguf", "pruned"]},
            "someone/Roleplay-Qwen3-8B-GGUF": {},
        }
        for model_id, extra in cases.items():
            with self.subTest(model_id):
                tags = extra.get("tags", ["gguf"])
                model = child(model_id, "Qwen/Qwen3-8B", tags=tags, relation=extra.get("relation", "quantized"))
                self.assertIsNotNone(rejection(model, "Qwen/Qwen3-8B", {}))
                self.assertEqual(rows_for(("Qwen/Qwen3-8B", model)), [])

    def test_untagged_official_conversions_count_by_name(self) -> None:
        untagged = {"id": "Qwen/Qwen3-8B-GGUF", "tags": ["gguf"], "downloads_all_time": 50}
        qat = child("Qwen/Qwen3-8B-qat-q4_0-gguf", "Qwen/Qwen3-8B", tags=["gguf"], downloads=10)
        self.assertEqual(
            [row["modelId"] for row in rows_for(("Qwen/Qwen3-8B", untagged), ("Qwen/Qwen3-8B", qat))],
            ["Qwen/Qwen3-8B-GGUF", "Qwen/Qwen3-8B-qat-q4_0-gguf"],
        )
        # A third party must still declare its base model.
        stray = {"id": "someone/Qwen3-8B-GGUF", "tags": ["gguf"], "downloads_all_time": 10**7}
        self.assertEqual(rejection(stray, "Qwen/Qwen3-8B", {}), "not_single_parent")

    def test_multi_parent_repositories_are_excluded(self) -> None:
        model = child("someone/Qwen3-8B-GGUF-2", "Qwen/Qwen3-8B", tags=["gguf", "base_model:Qwen/Qwen3-14B"])
        self.assertEqual(rejection(model, "Qwen/Qwen3-8B", {}), "not_single_parent")

    def test_admission_requires_official_publisher_or_lifetime_downloads(self) -> None:
        third_party = child("someone/Qwen3-8B-AWQ", "Qwen/Qwen3-8B", tags=["awq"], downloads=99_999)
        official = child("Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-8B", tags=["awq"], downloads=10)
        popular = child("other/Qwen3-8B-AWQ", "Qwen/Qwen3-8B", tags=["awq"], downloads=100_000)
        rows = rows_for(*(("Qwen/Qwen3-8B", m) for m in (third_party, official, popular)))
        self.assertEqual([row["modelId"] for row in rows], ["other/Qwen3-8B-AWQ", "Qwen/Qwen3-8B-AWQ"])

    def test_overrides_include_exclude_and_resolve_aliases(self) -> None:
        renamed = child(
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            tags=["gguf"],
        )
        rows = rows_for(
            ("meta-llama/Llama-3.1-8B-Instruct", renamed),
            ("", {"id": "someone/odd-name-awq", "tags": [], "downloads_all_time": 200_000}),
            ("Qwen/Qwen3-8B", child("someone/Qwen3-8B-GGUF", "Qwen/Qwen3-8B", tags=["gguf"])),
        )
        self.assertEqual(
            [(row["modelId"], row["parent_model_id"], row["local_formats"]) for row in rows],
            [
                ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "meta-llama/Llama-3.1-8B-Instruct", "gguf"),
                ("someone/odd-name-awq", "Qwen/Qwen3-14B", "awq"),
            ],
        )

    def test_catalog_is_deterministic_and_validates(self) -> None:
        candidates = [
            ("Qwen/Qwen3-8B", child("b/Qwen3-8B-GGUF", "Qwen/Qwen3-8B", tags=["gguf"])),
            ("Qwen/Qwen3-8B", child("A/Qwen3-8B-MLX", "Qwen/Qwen3-8B", library_name="mlx")),
        ]
        rows = rows_for(*candidates)
        self.assertEqual(rows, rows_for(*reversed(candidates)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local-models.csv"
            write_catalog(rows, path)
            self.assertEqual(len(validate_catalog(path, PARENTS, PARENTS, REGIONS)), 2)
            path.write_text(path.read_text().replace("china", "us", 1))
            with self.assertRaisesRegex(CatalogError, "parent_region"):
                validate_catalog(path, PARENTS, PARENTS, REGIONS)


if __name__ == "__main__":
    unittest.main()
