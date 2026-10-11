"""Public curation stays conversion-only, additive, and split by canonical scope."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.local_models import (
    CatalogError,
    curate_catalog,
    discover_local_models,
    main,
    read_catalog,
    validate_public_catalogs,
)


def parent(model_id="Qwen/Root", bucket="models"):
    org, name = model_id.split("/")
    return {"modelId": model_id, "org": org, "model": name, "parent_list": bucket}


def metadata(model_id="c/Root-GGUF", base="Qwen/Root", relation="quantized"):
    return {
        "id": model_id,
        "tags": ["gguf"],
        "base_model_ids": [base],
        "base_model_relation": relation,
    }


def discovery(raw=None, parents=None):
    return discover_local_models(
        raw or [metadata()], parents or [parent()], {"qwen": "china"}
    )


def correction(model_id, **fields):
    return {
        "model_id": model_id,
        "action": "exclude",
        "reason": "Reviewed correction",
        "source_urls": ["https://huggingface.co/c/Root-GGUF/blob/abcdef/README.md"],
        **fields,
    }


class PublicationTests(unittest.TestCase):
    def test_publication_excludes_modified_even_when_official_or_popular(self):
        rows = discovery(
            [
                metadata("Qwen/Root-GGUF", relation="finetune"),
                metadata("c/Root-GGUF", relation="pruned"),
            ]
        )["rows"]
        selected, report = curate_catalog(
            rows, [{"id": r["modelId"], "downloads": 1_000_000} for r in rows]
        )
        self.assertEqual(selected, [])
        self.assertEqual(report["conversion_candidates"], 0)

    def test_prior_admission_survives_low_or_missing_downloads(self):
        rows = discovery()["rows"]
        for metrics in (
            [],
            [{"id": rows[0]["modelId"], "downloads": 0, "downloads_all_time": 500}],
        ):
            with self.subTest(metrics=metrics):
                selected, report = curate_catalog(rows, metrics, rows)
                self.assertEqual(selected, rows)
                self.assertEqual(report["retained_below_threshold"], 1)
        self.assertEqual(curate_catalog(rows, [])[0], [])

    def test_prior_membership_does_not_bypass_missing_or_changed_source(self):
        old = discovery()["rows"]
        for current in ([], discovery([metadata(relation="finetune")])["rows"]):
            with (
                self.subTest(current=current),
                self.assertRaisesRegex(CatalogError, "Previously admitted"),
            ):
                curate_catalog(current, [], old)

    def test_reviewed_exclusion_or_category_correction_allows_removal(self):
        old = discovery()["rows"]
        self.assertEqual(
            curate_catalog([], [], old, [correction("c/Root-GGUF")])[0], []
        )
        changed = discovery([metadata(relation="finetune")])["rows"]
        reviewed = correction(
            "c/Root-GGUF",
            action="include",
            parent_model_id="Qwen/Root",
            modification_categories=["finetune"],
        )
        self.assertEqual(curate_catalog(changed, [], old, [reviewed])[0], [])

    def test_reviewed_exclusion_can_remove_a_repository_missing_from_metadata(self):
        old = discovery()["rows"]
        reviewed = [correction("c/Root-GGUF")]
        current = discover_local_models([], [parent()], {"qwen": "china"}, reviewed)
        self.assertEqual(curate_catalog(current["rows"], [], old, reviewed)[0], [])

    def test_unrelated_correction_cannot_authorize_removal(self):
        old = discovery()["rows"]
        with self.assertRaisesRegex(CatalogError, "Previously admitted"):
            curate_catalog([], [], old, [correction("unrelated/Repo")])

    def test_valid_parent_promotion_preserves_admission_and_changes_catalog(self):
        old = discovery(parents=[parent(bucket="extra_models")])["rows"]
        current = discovery()["rows"]
        selected, _ = curate_catalog(current, [], old)
        self.assertEqual(selected[0]["parent_list"], "models")
        validate_public_catalogs(selected, [], [parent()], {"qwen": "china"})

    def test_public_validator_rejects_wrong_scope_and_modified_rows(self):
        pure = discovery()["rows"]
        modified = discovery([metadata(relation="finetune")])["rows"]
        with self.assertRaisesRegex(CatalogError, "Wrong public catalog"):
            validate_public_catalogs([], pure, [parent()], {"qwen": "china"})
        with self.assertRaisesRegex(CatalogError, "modified or uncertain"):
            validate_public_catalogs(modified, [], [parent()], {"qwen": "china"})

    def test_old_research_modifications_are_not_retained_by_migration(self):
        rows = discovery([metadata(relation="finetune")])["rows"]
        self.assertEqual(curate_catalog([], [], rows)[0], [])

    def test_cli_splits_retains_replays_and_moves_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                key: root / name
                for key, name in {
                    "models": "models.csv",
                    "extra-models": "extra.csv",
                    "regions": "regions.csv",
                    "metadata": "input.jsonl",
                    "popularity": "popularity.jsonl",
                    "output": "local.csv",
                    "extra-output": "local-extra.csv",
                    "overrides": "overrides.json",
                    "aliases": "aliases.json",
                    "evidence": "evidence.jsonl",
                    "report": "report.json",
                }.items()
            }
            paths["models"].write_text("org,model,modelId\nQwen,Root,Qwen/Root\n")
            paths["extra-models"].write_text(
                "org,model,modelId\nQwen,Other,Qwen/Other\n"
            )
            paths["regions"].write_text("hf_org,region,notes\nQwen,china,\n")
            raw = [
                metadata(),
                metadata("c/Other-GGUF", "Qwen/Other"),
                metadata("c/Root-Finetune-GGUF", relation="finetune"),
            ]
            paths["metadata"].write_text("".join(json.dumps(row) + "\n" for row in raw))
            paths["popularity"].write_text(
                "".join(
                    json.dumps({"id": row["id"], "downloads": 10_000}) + "\n"
                    for row in raw
                )
            )
            args = [
                value
                for key, path in paths.items()
                for value in ("--" + key, str(path))
            ]
            main(args)
            self.assertEqual(
                [r["modelId"] for r in read_catalog(paths["output"])], ["c/Root-GGUF"]
            )
            self.assertEqual(
                [r["modelId"] for r in read_catalog(paths["extra-output"])],
                ["c/Other-GGUF"],
            )
            # Usage declines; retained candidates still need freshly validated lineage.
            paths["popularity"].write_text("")
            main(args)
            main([*args, "--check"])
            main([*args, "--validate"])
            self.assertEqual(
                json.loads(paths["report"].read_text())["curation"][
                    "retained_below_threshold"
                ],
                2,
            )
            # Canonical promotion moves the backup row into the default catalog.
            paths["models"].write_text(
                paths["models"].read_text() + "Qwen,Other,Qwen/Other\n"
            )
            paths["extra-models"].write_text("org,model,modelId\n")
            main(args)
            self.assertEqual(len(read_catalog(paths["output"])), 2)
            self.assertEqual(read_catalog(paths["extra-output"]), [])
            # Incomplete metadata cannot silently erase a previously admitted conversion.
            paths["metadata"].write_text(json.dumps(raw[0]) + "\n")
            before = paths["output"].read_bytes(), paths["extra-output"].read_bytes()
            with self.assertRaisesRegex(CatalogError, "Previously admitted"):
                main(args)
            self.assertEqual(
                before,
                (paths["output"].read_bytes(), paths["extra-output"].read_bytes()),
            )
