import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.local_models import (
    CatalogError,
    candidate_csv_bytes,
    csv_bytes,
    discover_local_models,
    file_sha,
    format_evidence,
    iter_metadata,
    jsonl_bytes,
    main,
    name_corroboration,
    normalize_metadata,
    validate_catalog,
    validate_catalog_rows,
)


def metadata(
    model_id, *, base=None, formats=("gguf",), relation="quantized", card=None, **kwargs
):
    return {
        "id": model_id,
        "tags_json": list(formats),
        "base_model_ids_json": [base] if base else [],
        "base_model_relation": relation,
        "card_data_json": card or {},
        **kwargs,
    }


def parent(model_id="Qwen/Root", parent_list="models"):
    org, model = model_id.split("/")
    return {"org": org, "model": model, "modelId": model_id, "parent_list": parent_list}


def discover(rows, parents=None, overrides=None):
    return discover_local_models(
        rows, parents or [parent()], {"qwen": "china", "meta": "us"}, overrides
    )


def override(model_id, parent_id=None, **kwargs):
    return {
        "model_id": model_id,
        "action": "include" if parent_id else "exclude",
        **({"parent_model_id": parent_id} if parent_id else {}),
        "reason": "Reviewed conversion statement",
        "source_urls": ["https://huggingface.co/org/repo/blob/abcdef/README.md"],
        **kwargs,
    }


class LocalModelsTests(unittest.TestCase):
    def test_direct_parent_and_metadata_join(self):
        result = discover([metadata("converter/Root-GGUF", base="qwen/root")])
        row = result["rows"][0]
        self.assertEqual(row["parent_model_id"], "Qwen/Root")
        self.assertEqual(row["parent_region"], "china")
        self.assertEqual(row["parent_list"], "models")
        self.assertEqual(row["base_model_id"], "Qwen/Root")
        self.assertEqual(row["mapping_method"], "direct")
        validate_catalog(
            result["rows"], result["evidence"], [parent()], {"qwen": "china"}
        )

    def test_extra_parent_and_unknown_region(self):
        result = discover(
            [metadata("c/Other-MLX", base="Extra/Other", formats=("mlx",))],
            [parent("Extra/Other", "extra_models")],
        )
        self.assertEqual(result["rows"][0]["parent_list"], "extra_models")
        self.assertEqual(result["rows"][0]["parent_region"], "unknown")

    def test_reviewed_redirects_coalesce_aliases_without_guessing_parent_names(self):
        canonical = "Qwen/Root"
        aliases = [
            {
                "alias_id": "Former/Root",
                "canonical_id": canonical,
                "reason": "Reviewed HTTP repository redirect",
                "http_status": 307,
                "redirect_url": "https://huggingface.co/api/models/Qwen/Root",
                "source_urls": ["https://huggingface.co/api/models/Former/Root"],
                "observed_at": "2026-10-10T00:00:00Z",
                "resolved_sha": "a" * 40,
            }
        ]
        source = metadata("c/Root-GGUF", base=canonical)
        source["base_model_ids_json"].append("Former/Root")
        self.assertEqual(
            discover([source])["candidates"][0]["reason"], "ambiguous_parent"
        )
        result = discover_local_models(
            [source], [parent()], {"qwen": "china"}, aliases=aliases
        )
        self.assertEqual(result["rows"][0]["base_model_id"], canonical)
        self.assertEqual(
            len(result["evidence"][0]["base_evidence"]["declared_base_model_ids"]), 2
        )
        validate_catalog(
            result["rows"],
            result["evidence"],
            [parent()],
            {"qwen": "china"},
            aliases=aliases,
        )
        with self.assertRaises(CatalogError):
            validate_catalog(
                result["rows"], result["evidence"], [parent()], {"qwen": "china"}
            )
        with self.assertRaises(CatalogError):
            discover_local_models(
                [source], [parent()], {}, aliases=[{**aliases[0], "http_status": 200}]
            )

    def test_reviewed_modification_propagates_and_cannot_be_hidden_in_csv(self):
        reviewed = [
            override(
                "a/Renamed-GPTQ", "Qwen/Root", modification_categories=["finetune"]
            )
        ]
        result = discover(
            [
                metadata("a/Renamed-GPTQ", formats=("gptq",)),
                metadata(
                    "b/Renamed-GGUF",
                    base="a/Renamed-GPTQ",
                    card={"base_model_relation": "quantized"},
                ),
            ],
            overrides=reviewed,
        )
        self.assertEqual(result["rows"][1]["variant_category"], "finetune")
        validate_catalog(
            result["rows"], result["evidence"], [parent()], {"qwen": "china"}, reviewed
        )
        result["rows"][1]["is_modified"] = "false"
        with self.assertRaisesRegex(CatalogError, "modification categories"):
            validate_catalog(
                result["rows"],
                result["evidence"],
                [parent()],
                {"qwen": "china"},
                reviewed,
            )

    def test_multiformat_and_artifact_evidence_are_not_deduplicated_to_one_format(self):
        row = metadata(
            "c/Root-quantized",
            base="Qwen/Root",
            formats=("gguf", "mlx"),
            siblings_json=["b.gguf", "a.gguf", "onnx/model.onnx", "openvino_model.xml"],
        )
        result = discover([row])
        self.assertEqual(result["rows"][0]["local_formats"], "gguf;mlx;onnx;openvino")
        files = [
            item
            for item in result["evidence"][0]["format_evidence"]
            if item["kind"] == "file" and item["format"] == "gguf"
        ]
        self.assertEqual(
            files,
            [
                {
                    "format": "gguf",
                    "kind": "file",
                    "value": "a.gguf",
                    "matching_files": 2,
                }
            ],
        )

    def test_config_quantization_and_aliases(self):
        row = normalize_metadata(
            metadata(
                "c/Root",
                base="Qwen/Root",
                formats=("auto-gptq", "mlx-vlm"),
                config_json={"quantization_config": {"quant_method": "aqlm"}},
            )
        )
        self.assertEqual(
            {item["format"] for item in format_evidence(row)}, {"gptq", "mlx", "aqlm"}
        )
        runtime = normalize_metadata(
            metadata("c/Root", base="Qwen/Root", formats=("exllamav2",))
        )
        self.assertEqual(format_evidence(runtime), [])

    def test_names_only_never_auto_include(self):
        result = discover([metadata("c/Root-GGUF", base="Qwen/Root", formats=())])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["candidates"][0]["reason"], "name_only_format_hint")
        self.assertTrue(result["candidates"][0]["tracked_association"])

    def test_divergent_names_are_unknown_except_explicit_publisher_conversion(self):
        row = metadata("c/Renamed-GGUF", base="Qwen/Root")
        self.assertEqual(discover([row])["rows"][0]["variant_category"], "unknown")
        row["card_data_json"] = {"base_model_relation": "quantized"}
        result = discover([row])
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(
            result["evidence"][0]["base_evidence"]["relation_provenance"],
            "publisher_card",
        )

    def test_training_modifiers_outside_parent_are_not_conversions(self):
        for modifier in (
            "dpo",
            "abliterated",
            "obliterated",
            "lora",
            "distilled",
            "uncensored",
        ):
            with self.subTest(modifier=modifier):
                result = discover(
                    [
                        metadata(
                            f"c/Root-{modifier}-GGUF",
                            base="Qwen/Root",
                            card={"base_model_relation": "quantized"},
                        )
                    ]
                )
                self.assertEqual(result["rows"][0]["is_modified"], "true")
                self.assertNotEqual(result["rows"][0]["variant_category"], "conversion")

    def test_training_words_inside_tracked_parent_do_not_reject_conversion(self):
        checkpoint = "Qwen/Root-Distilled-DPO"
        result = discover(
            [metadata("c/Root-Distilled-DPO-GGUF", base=checkpoint)],
            [parent(checkpoint)],
        )
        self.assertEqual(len(result["rows"]), 1)

    def test_multihop_conversion_chain_preserves_original_parent(self):
        result = discover(
            [
                metadata("a/Root-GPTQ", base="Qwen/Root", formats=("gptq",)),
                metadata("b/Root-GGUF", base="a/Root-GPTQ"),
            ]
        )
        row = next(row for row in result["rows"] if row["org"] == "b")
        self.assertEqual(row["parent_model_id"], "Qwen/Root")
        self.assertEqual(row["base_model_id"], "a/Root-GPTQ")
        self.assertEqual(row["mapping_method"], "lineage_chain")
        evidence = next(
            row for row in result["evidence"] if row["model_id"].startswith("b/")
        )
        self.assertEqual(
            evidence["lineage"], ["b/Root-GGUF", "a/Root-GPTQ", "Qwen/Root"]
        )
        self.assertEqual(
            evidence["intermediate_evidence"][0]["model_id"], "a/Root-GPTQ"
        )

    def test_finetune_adapter_and_single_parent_merge_flags_propagate(self):
        for relation in ("finetune", "adapter", "merge"):
            with self.subTest(relation=relation):
                result = discover(
                    [
                        metadata(
                            "a/Root-GPTQ",
                            base="Qwen/Root",
                            formats=("gptq",),
                            relation=relation,
                        ),
                        metadata("b/Root-GGUF", base="a/Root-GPTQ"),
                    ]
                )
                self.assertEqual(len(result["rows"]), 2)
                self.assertTrue(
                    all(row["variant_category"] == relation for row in result["rows"])
                )
                self.assertEqual(
                    result["rows"][1]["lineage_categories"], "conversion;" + relation
                )

    def test_tracked_mixed_original_and_tracked_finetune_are_terminal_parents(self):
        checkpoint = "Qwen/Root-Instruct"
        result = discover(
            [
                metadata(
                    checkpoint, base="Qwen/Root", formats=("onnx",), relation="finetune"
                ),
                metadata("c/Root-Instruct-GGUF", base=checkpoint),
            ],
            [parent(), parent(checkpoint, "extra_models")],
        )
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["parent_model_id"], checkpoint)
        self.assertEqual(result["rows"][0]["parent_list"], "extra_models")
        self.assertEqual(result["candidates"][0]["reason"], "tracked_original")

    def test_tracked_actual_conversion_is_flagged_and_chain_continues(self):
        result = discover(
            [
                metadata("Qwen/Root-GPTQ", base="Qwen/Root", formats=("gptq",)),
                metadata("c/Root-GGUF", base="Qwen/Root-GPTQ"),
            ],
            [parent(), parent("Qwen/Root-GPTQ")],
        )
        tracked = next(
            row for row in result["rows"] if row["modelId"] == "Qwen/Root-GPTQ"
        )
        self.assertEqual(tracked["already_tracked"], "true")
        self.assertEqual(tracked["parent_model_id"], "Qwen/Root")
        self.assertEqual(result["rows"][0]["parent_model_id"], "Qwen/Root")

    def test_cycle_and_missing_parent_are_reported(self):
        cycle = discover(
            [
                metadata("a/Root-GGUF", base="b/Root-GGUF"),
                metadata("b/Root-GGUF", base="a/Root-GGUF"),
            ]
        )
        self.assertEqual(cycle["rows"], [])
        self.assertEqual(
            {row["reason"] for row in cycle["candidates"]}, {"lineage_cycle"}
        )
        missing = discover([metadata("c/Root-GGUF", base="Unknown/Root")])
        self.assertEqual(missing["candidates"][0]["reason"], "missing_parent_metadata")

    def test_excessive_lineage_depth_does_not_crash_catalog(self):
        rows = [
            metadata(f"x/{index}-Root-GGUF", base=f"x/{index + 1}-Root-GGUF")
            for index in range(80)
        ]
        rows.append(metadata("x/80-Root-GGUF", base="Qwen/Root"))
        result = discover(rows)
        self.assertIn(
            "lineage_depth_exceeded", result["report"]["unresolved_by_reason"]
        )

    def test_ambiguous_and_invalid_parent_metadata_do_not_abort_other_repos(self):
        bad = metadata("c/Root-GGUF", base="Qwen/Root")
        bad["tags_json"].append("base_model:Meta/Other")
        result = discover(
            [bad, metadata("c/Root-MLX", base="Qwen/Root", formats=("mlx",))]
        )
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["candidates"][0]["reason"], "ambiguous_parent")
        bad["base_model_ids_json"] = ["https://huggingface.co/Qwen/Root"]
        self.assertEqual(
            discover([bad])["candidates"][0]["reason"], "invalid_parent_evidence"
        )

    def test_private_and_disabled_intermediates_are_rejected(self):
        for flag in ("private", "disabled"):
            rows = [
                metadata(
                    "a/Root-GPTQ", base="Qwen/Root", formats=("gptq",), **{flag: True}
                ),
                metadata("c/Root-GGUF", base="a/Root-GPTQ"),
            ]
            self.assertEqual(discover(rows)["rows"], [])

    def test_reviewed_parent_correction_preserves_conflicting_metadata_evidence(self):
        rows = [metadata("c/Root-Instruct-MLX", base="Qwen/Root", formats=("mlx",))]
        reviewed = [override("c/Root-Instruct-MLX", "Qwen/Root-Instruct")]
        parents = [parent(), parent("Qwen/Root-Instruct")]
        result = discover(rows, parents, reviewed)
        self.assertEqual(result["rows"][0]["parent_model_id"], "Qwen/Root-Instruct")
        self.assertEqual(result["rows"][0]["base_model_id"], "Qwen/Root-Instruct")
        self.assertEqual(
            result["evidence"][0]["base_evidence"]["base_model_ids"], ["Qwen/Root"]
        )
        self.assertEqual(result["rows"][0]["mapping_method"], "reviewed_override")
        validate_catalog(
            result["rows"], result["evidence"], parents, {"qwen": "china"}, reviewed
        )

    def test_reviewed_missing_parent_and_format_and_exclusion(self):
        row = metadata("c/Renamed", formats=(), relation=None)
        reviewed = [override("c/Renamed", "Qwen/Root", local_formats=["gguf"])]
        self.assertEqual(len(discover([row], overrides=reviewed)["rows"]), 1)
        self.assertEqual(
            discover(
                [metadata("c/Root-GGUF", base="Qwen/Root")],
                overrides=[override("c/Root-GGUF")],
            )["rows"],
            [],
        )
        with self.assertRaises(CatalogError):
            discover(
                [row], overrides=[override("c/Renamed", "Qwen/Root", source_urls=[])]
            )
        result = discover([row], overrides=reviewed)
        self.assertEqual(result["rows"][0]["base_model_id"], "Qwen/Root")
        validate_catalog(
            result["rows"], result["evidence"], [parent()], {"qwen": "china"}, reviewed
        )

    def test_metadata_duplicates_conflicts_and_deterministic_order(self):
        a, b = (
            metadata("z/Root-GGUF", base="Qwen/Root"),
            metadata("A/Root-GGUF", base="Qwen/Root"),
        )
        first = discover([a, b, a])
        second = discover([b, a])
        self.assertEqual(first, second)
        self.assertEqual([row["org"] for row in first["rows"]], ["A", "z"])
        self.assertNotIn(b"\r", csv_bytes(first["rows"]))
        with self.assertRaises(CatalogError):
            discover([a, {**a, "base_model_ids_json": ["Meta/Other"]}])

    def test_compressed_evidence_is_deterministic_and_lossless(self):
        evidence = discover([metadata("c/Root-GGUF", base="Qwen/Root")])["evidence"]
        compressed = jsonl_bytes(evidence, compressed=True)
        self.assertEqual(compressed, jsonl_bytes(evidence, compressed=True))
        self.assertEqual(gzip.decompress(compressed), jsonl_bytes(evidence))
        self.assertEqual(compressed[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(compressed[3] & 8, 0)  # No original filename header.

    def test_metadata_native_and_encoded_fields_agree_and_creation_is_not_release(self):
        native = metadata(
            "c/Root-GGUF", base="Qwen/Root", created_at="2024-01-01T23:30:00-02:00"
        )
        encoded = {
            **native,
            "tags_json": json.dumps(native["tags_json"]),
            "base_model_ids_json": json.dumps(native["base_model_ids_json"]),
            "card_data_json": "null",
        }
        self.assertEqual(discover([native])["rows"], discover([encoded])["rows"])
        self.assertEqual(discover([native])["rows"][0]["created_at"], "2024-01-02")

    def test_validation_rejects_bad_parent_join_evidence_and_formats(self):
        result = discover([metadata("c/Root-GGUF", base="Qwen/Root")])
        result["rows"][0]["parent_region"] = "us"
        with self.assertRaises(CatalogError):
            validate_catalog(
                result["rows"], result["evidence"], [parent()], {"qwen": "china"}
            )
        result["rows"][0]["parent_region"] = "china"
        result["evidence"][0]["parent_model_id"] = "Wrong/Root"
        with self.assertRaises(CatalogError):
            validate_catalog(
                result["rows"], result["evidence"], [parent()], {"qwen": "china"}
            )

    def test_cli_replay_and_source_free_validation_preserve_primary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models.csv"
            extra = root / "extra_models.csv"
            regions = root / "regions.csv"
            meta = root / "metadata.jsonl"
            output, evidence, report = (
                root / name
                for name in (
                    "local.csv",
                    ".local-model-cache/local_models_evidence.jsonl.gz",
                    ".local-model-cache/local_models_report.json",
                )
            )
            models.write_text("org,model,modelId\nQwen,Root,Qwen/Root\n")
            extra.write_text("org,model,modelId\n")
            regions.write_text("hf_org,region,notes\nQwen,china,\n")
            meta.write_text(
                json.dumps(metadata("c/Root-GGUF", base="Qwen/Root")) + "\n"
            )
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "sha256": file_sha(meta), "rows": 1})
            )
            before = models.read_bytes(), extra.read_bytes()
            args = [
                "--models",
                str(models),
                "--extra-models",
                str(extra),
                "--regions",
                str(regions),
                "--output",
                str(output),
                "--overrides",
                str(root / "absent-overrides.json"),
                "--aliases",
                str(root / "absent-aliases.json"),
                "--source-manifest",
                str(manifest),
            ]
            with patch("scripts.local_models.ROOT", root):
                main([*args, "--metadata", str(meta)])
                main([*args, "--metadata", str(meta), "--check"])
                self.assertTrue(evidence.exists())
                self.assertTrue(report.exists())
                original_manifest = manifest.read_bytes()
                manifest.write_text(
                    json.dumps({"schema_version": 1, "sha256": "0" * 64, "rows": 1})
                )
                with self.assertRaisesRegex(CatalogError, "Source manifest"):
                    main([*args, "--metadata", str(meta), "--check"])
                manifest.write_bytes(original_manifest)
            self.assertEqual(before, (models.read_bytes(), extra.read_bytes()))
            # Plain CSV validation has no generated evidence/report/input dependency.
            for path in (meta, manifest, evidence, report):
                path.unlink()
            main([*args[:-2], "--validate"])
            # Unrelated weekly canonical additions do not stale a valid catalog.
            models.write_text(models.read_text() + "Qwen,Another,Qwen/Another\n")
            main([*args[:-2], "--validate"])
            output.write_bytes(output.read_bytes().replace(b"china", b"us"))
            with self.assertRaises(CatalogError):
                main([*args[:-2], "--validate"])

    def test_csv_only_validation_checks_parent_identity_formats_and_lineage(self):
        row = discover([metadata("c/Root-GGUF", base="Qwen/Root")])["rows"][0]
        validate_catalog_rows([row], [parent()], {"qwen": "china"})
        invalid = [
            {"parent_model_id": "qwen/root"},
            {"parent_org": "c"},
            {"parent_model": "Other"},
            {"parent_list": "extra_models"},
            {"parent_region": "us"},
            {"org": "wrong"},
            {"local_formats": "mlx;gguf"},
            {"local_formats": "gguf;gguf"},
            {"local_formats": "invented"},
            {"already_tracked": "true"},
            {"created_at": "2026-02-30"},
            {"base_model_id": "Other/Root"},
            {"parent_lineage": "c/Root-GGUF;Qwen/Root;c/Root-GGUF;Qwen/Root"},
            {"parent_lineage": "Other/Root;Qwen/Root"},
            {"mapping_method": "lineage_chain"},
            {"model": None},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(CatalogError):
                validate_catalog_rows(
                    [{**row, **changes}], [parent()], {"qwen": "china"}
                )
        with self.assertRaisesRegex(CatalogError, "Duplicate"):
            validate_catalog_rows([row, row], [parent()], {"qwen": "china"})
        other = discover([metadata("a/Root-GGUF", base="Qwen/Root")])["rows"][0]
        with self.assertRaisesRegex(CatalogError, "sorted"):
            validate_catalog_rows([row, other], [parent()], {"qwen": "china"})

    def test_csv_only_validation_checks_category_aggregation(self):
        result = discover(
            [
                metadata(
                    "c/Root-GGUF",
                    base="middle/Root-SFT",
                    card={"base_model_relation": "quantized"},
                ),
                metadata(
                    "middle/Root-SFT", base="Qwen/Root", formats=(), relation="finetune"
                ),
            ]
        )
        row = result["rows"][0]
        validate_catalog_rows([row], [parent()], {"qwen": "china"})
        for changes in [
            {"is_modified": "false"},
            {"variant_category": "conversion"},
            {"modification_categories": ""},
            {"lineage_categories": "conversion"},
            {"lineage_categories": "conversion;conversion+finetune"},
            {"lineage_categories": "conversion;finetune+finetune"},
            {"lineage_categories": "conversion;invented"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(CatalogError):
                validate_catalog_rows(
                    [{**row, **changes}], [parent()], {"qwen": "china"}
                )
        # Structural checks cannot establish that declared source categories are true.
        consistent_but_unproved = {
            **row,
            "lineage_categories": "conversion;conversion",
            "is_modified": "false",
            "variant_category": "conversion",
            "modification_categories": "",
        }
        validate_catalog_rows([consistent_but_unproved], [parent()], {"qwen": "china"})
        with self.assertRaisesRegex(CatalogError, "categories"):
            validate_catalog(
                [consistent_but_unproved],
                result["evidence"],
                [parent()],
                {"qwen": "china"},
            )

    def test_csv_only_validation_enforces_reviewed_intermediate_corrections(self):
        reviewed = override(
            "middle/Changed", "Qwen/Root", modification_categories=["pruned"]
        )
        result = discover(
            [
                metadata(
                    "c/Changed-GGUF",
                    base="middle/Changed",
                    card={"base_model_relation": "quantized"},
                ),
                metadata("middle/Changed", formats=()),
            ],
            overrides=[reviewed],
        )
        row = result["rows"][0]
        validate_catalog_rows([row], [parent()], {"qwen": "china"}, [reviewed])
        for replacement in [
            override("middle/Changed"),
            {**reviewed, "base_model_id": "other/Root"},
            {**reviewed, "modification_categories": ["finetune"]},
        ]:
            with self.subTest(replacement=replacement), self.assertRaises(CatalogError):
                validate_catalog_rows(
                    [row], [parent()], {"qwen": "china"}, [replacement]
                )
        leaf = override("c/Renamed", "Qwen/Root", local_formats=["gguf"])
        leaf_row = discover([metadata("c/Renamed")], overrides=[leaf])["rows"][0]
        with self.assertRaisesRegex(CatalogError, "matching override"):
            validate_catalog_rows([leaf_row], [parent()], {"qwen": "china"})
        with self.assertRaisesRegex(CatalogError, "formats"):
            validate_catalog_rows(
                [{**leaf_row, "local_formats": "mlx"}],
                [parent()],
                {"qwen": "china"},
                [leaf],
            )

    def test_standard_conversion_configuration_suffixes_are_accepted(self):
        cases = [
            ("Root-IQ4_NL-GGUF", "gguf"),
            ("Root-W4A16-actorder-true-GPTQ", "gptq"),
            ("Root-1x16-aqlm", "aqlm"),
            ("Root-AutoRound-MXFP4-GGUF-sharded", "gguf"),
            ("Root-NVFP4-MLX", "mlx"),
            ("Root-NF4-bnb", "bitsandbytes"),
        ]
        for name, fmt in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    len(
                        discover(
                            [metadata("c/" + name, base="Qwen/Root", formats=(fmt,))]
                        )["rows"]
                    ),
                    1,
                )

    def test_delivery_spelling_aliases_keep_explicit_parent_identity(self):
        for model_id, parent_id in (
            ("TheBloke/Llama-2-7B-Chat-GGUF", "meta-llama/Llama-2-7b-chat-hf"),
            ("unsloth/llama-3-8b-bnb-4bit", "meta-llama/Meta-Llama-3-8B"),
            ("c/OLMo-7B-SFT-GGUF", "allenai/OLMo-7B-SFT-hf"),
        ):
            with self.subTest(model=model_id):
                result = discover(
                    [metadata(model_id, base=parent_id)], [parent(parent_id)]
                )
                self.assertEqual(result["rows"][0]["parent_model_id"], parent_id)
                self.assertEqual(result["rows"][0]["base_model_id"], parent_id)
                validate_catalog(
                    result["rows"], result["evidence"], [parent(parent_id)], {}
                )
        self.assertFalse(
            name_corroboration("c/Llama-3-8B-GGUF", "MetaResearch/Meta-Llama-3-8B")[0]
        )
        self.assertFalse(
            name_corroboration(
                "c/Llama-2-7B-Chat-GGUF", "meta-llama/Llama-2-13b-chat-hf"
            )[0]
        )

    def test_pruning_and_weight_modifications_do_not_bypass_explicit_quantized_card(
        self,
    ):
        for modifier in ("REAP25", "pruned", "pruning", "heretic"):
            with self.subTest(modifier=modifier):
                row = metadata(
                    "c/Root-" + modifier + "-GGUF",
                    base="Qwen/Root",
                    card={"base_model_relation": "quantized"},
                )
                self.assertEqual(discover([row])["rows"][0]["is_modified"], "true")
                tracked = "Qwen/Root-" + modifier
                self.assertEqual(
                    len(
                        discover(
                            [metadata("c/Root-" + modifier + "-GGUF", base=tracked)],
                            [parent(tracked)],
                        )["rows"]
                    ),
                    1,
                )

    def test_malformed_parent_ids_are_reviewed_without_index_errors(self):
        self.assertFalse(name_corroboration("c/Root-GGUF", "x/--")[0])
        for bad in (
            "x/--",
            "https://huggingface.co/Qwen/Root",
            "unqualified",
            "Qwen/Root with spaces",
        ):
            with self.subTest(parent=bad):
                result = discover([metadata("c/Root-GGUF", base=bad)])
                self.assertEqual(
                    result["candidates"][0]["reason"], "invalid_parent_evidence"
                )

    def test_reviewed_intermediate_chain_carries_its_review_evidence(self):
        rows = [
            metadata("a/Renamed-GPTQ", formats=("gptq",)),
            metadata("c/Root-GGUF", base="a/Renamed-GPTQ"),
        ]
        reviewed = [override("a/Renamed-GPTQ", "Qwen/Root")]
        result = discover(rows, overrides=reviewed)
        validate_catalog(
            result["rows"], result["evidence"], [parent()], {"qwen": "china"}, reviewed
        )
        child = next(
            item for item in result["evidence"] if item["model_id"].startswith("c/")
        )
        self.assertEqual(child["intermediate_evidence"][0]["override"], reviewed[0])
        child["intermediate_evidence"] = []
        with self.assertRaises(CatalogError):
            validate_catalog(
                result["rows"],
                result["evidence"],
                [parent()],
                {"qwen": "china"},
                reviewed,
            )

    def test_source_free_validation_checks_format_and_immediate_base_proof(self):
        result = discover([metadata("c/Root-GGUF", base="Qwen/Root")])
        result["evidence"][0]["format_evidence"][0]["value"] = "not-a-format"
        with self.assertRaises(CatalogError):
            validate_catalog(
                result["rows"], result["evidence"], [parent()], {"qwen": "china"}
            )
        result = discover([metadata("c/Root-GGUF", base="Qwen/Root")])
        result["rows"][0]["base_model_id"] = "Meta/Wrong"
        with self.assertRaises(CatalogError):
            validate_catalog(
                result["rows"], result["evidence"], [parent()], {"qwen": "china"}
            )

    def test_metadata_and_jsonl_adapter_ignore_unneeded_large_config(self):
        row = metadata(
            "c/Root-GGUF",
            base="Qwen/Root",
            config_json={"irrelevant_tensor": "x" * 10000},
            card_data_json={"license": "mit"},
        )
        self.assertEqual(normalize_metadata(row)["config"], {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(list(iter_metadata(path)), [row])

    def test_parquet_adapter_matches_jsonl_discovery(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest("Optional Parquet adapter requires DuckDB")
        source = metadata(
            "c/Root-GGUF",
            base="Qwen/Root",
            siblings_json=["model.gguf"],
            created_at="2026-01-02T03:04:05Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "source.jsonl"
            parquet = Path(directory) / "source.parquet"
            jsonl.write_text(json.dumps(source) + "\n")
            connection = duckdb.connect()
            try:
                connection.execute(
                    "CREATE TABLE source AS SELECT * FROM read_json_auto(?)",
                    [str(jsonl)],
                )
                connection.execute("COPY source TO ? (FORMAT PARQUET)", [str(parquet)])
            finally:
                connection.close()
            self.assertEqual(
                discover(iter_metadata(jsonl)), discover(iter_metadata(parquet))
            )

    def test_report_lists_uncovered_parents_without_claiming_no_variants_exist(self):
        result = discover(
            [metadata("c/Root-GGUF", base="Qwen/Root")],
            [parent(), parent("Meta/Other", "extra_models")],
        )
        self.assertEqual(
            result["report"]["parent_coverage"]["models"][
                "parents_with_observed_variants"
            ],
            1,
        )
        self.assertEqual(
            result["report"]["parent_coverage"]["extra_models"][
                "parents_without_observed_variants"
            ],
            ["Meta/Other"],
        )

    def test_compact_candidate_csv_preserves_review_identifiers_and_reasons(self):
        result = discover([metadata("c/Root-GGUF", base="Qwen/Root", formats=())])
        payload = candidate_csv_bytes(result["candidates"])
        self.assertEqual(
            payload.decode().splitlines()[0],
            "model_id,local_formats,reason,base_model_ids,lineage",
        )
        self.assertIn("name_only_format_hint", payload.decode())
        self.assertIn("Qwen/Root", payload.decode())
        self.assertNotIn(b"\r", payload)

    def test_cli_refuses_to_overwrite_primary_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models, extra, regions = (
                root / name for name in ("models.csv", "extra.csv", "regions.csv")
            )
            models.write_text("org,model,modelId\nQwen,Root,Qwen/Root\n")
            extra.write_text("org,model,modelId\n")
            regions.write_text("hf_org,region,notes\nQwen,china,\n")
            before = models.read_bytes()
            with self.assertRaises(CatalogError):
                main(
                    [
                        "--models",
                        str(models),
                        "--extra-models",
                        str(extra),
                        "--regions",
                        str(regions),
                        "--output",
                        str(models),
                    ]
                )
            self.assertEqual(models.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
