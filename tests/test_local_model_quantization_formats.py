"""Regression coverage for formats found by the global popularity audit."""

import unittest

from scripts.local_models import discover_local_models, validate_catalog


class PopularQuantizationFormatsTests(unittest.TestCase):
    def test_modelopt_requires_explicit_format_evidence(self):
        parents = [
            {
                "org": "Qwen",
                "model": "Root",
                "modelId": "Qwen/Root",
                "parent_list": "models",
            }
        ]
        for evidence in (
            {"tags": ["modelopt"]},
            {"config": {"quantization_config": {"quant_method": "modelopt"}}},
        ):
            with self.subTest(evidence=evidence):
                source = {
                    "id": "nvidia/Root-NVFP4",
                    "base_model_ids": ["Qwen/Root"],
                    "card_data": {"base_model_relation": "quantized"},
                    **evidence,
                }
                result = discover_local_models([source], parents, {"qwen": "china"})
                self.assertEqual(result["rows"][0]["local_formats"], "modelopt")
                validate_catalog(
                    result["rows"], result["evidence"], parents, {"qwen": "china"}
                )
        source.pop("config")
        source["tags"] = ["nvfp4"]
        self.assertEqual(
            discover_local_models([source], parents, {"qwen": "china"})["rows"], []
        )

    def test_autoround_aliases_are_real_format_evidence(self):
        parents = [
            {
                "org": "Qwen",
                "model": "Root",
                "modelId": "Qwen/Root",
                "parent_list": "models",
            }
        ]
        for alias in ("auto-round", "autoround", "auto_round"):
            for evidence in (
                {"tags": [alias]},
                {"config": {"quantization_config": {"quant_method": alias, "bits": 4}}},
            ):
                with self.subTest(alias=alias, evidence=evidence):
                    source = {
                        "id": "converter/Root-int4-AutoRound",
                        "base_model_ids": ["Qwen/Root"],
                        "card_data": {"base_model_relation": "quantized"},
                        **evidence,
                    }
                    result = discover_local_models([source], parents, {"qwen": "china"})
                    row = result["rows"][0]
                    self.assertEqual(row["local_formats"], "auto-round")
                    self.assertEqual(row["is_modified"], "false")
                    self.assertEqual(row["parent_model_id"], "Qwen/Root")
                    validate_catalog(
                        result["rows"], result["evidence"], parents, {"qwen": "china"}
                    )

    def test_autoround_name_alone_is_not_format_evidence(self):
        parents = [
            {
                "org": "Qwen",
                "model": "Root",
                "modelId": "Qwen/Root",
                "parent_list": "models",
            }
        ]
        source = {"id": "converter/Root-AutoRound", "base_model_ids": ["Qwen/Root"]}
        result = discover_local_models([source], parents, {"qwen": "china"})
        self.assertEqual(result["rows"], [])


if __name__ == "__main__":
    unittest.main()
