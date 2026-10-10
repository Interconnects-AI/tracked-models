import importlib.util
import json
import tempfile
import unittest
import tracemalloc
import ijson
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("collect_local_models", Path(__file__).resolve().parents[1] / "scripts" / "collect_local_models.py")
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)


class CollectorTests(unittest.TestCase):
    def info(self, **kwargs):
        return SimpleNamespace(id="pub/variant", private=False, disabled=False,
                               card_data={"base_model": "origin/model"},
                               tags=["gguf", "gguf"], siblings=[SimpleNamespace(rfilename="a.gguf")],
                               created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), **kwargs)

    def test_public_normalization_keeps_parent_and_file_evidence(self):
        row = collect.normalize_model(self.info(), "2026-10-10")
        self.assertEqual(row["base_model_ids_json"], ["origin/model"])
        self.assertEqual(row["siblings_json"], ["a.gguf"])
        self.assertEqual(row["tags_json"], ["gguf"])
        self.assertTrue(row["created_at"].endswith("+00:00"))

    def test_private_metadata_never_enters_output_but_disabled_tombstones_remain(self):
        info = self.info()
        info.private = True
        self.assertIsNone(collect.normalize_model(info, "now"))
        info.private = False
        info.disabled = True
        self.assertTrue(collect.normalize_model(info, "now")["disabled"])

    def test_compact_card_evidence_is_preserved_without_chat_template_bulk(self):
        info = self.info(config={"quantization_config": {"quant_method": "awq"}, "chat_template": "bulk"})
        info.card_data.update({"tags": ["awq"], "library_name": "mlx", "quantization_config": {"bits": 4}, "model-index": "bulk"})
        row = collect.normalize_model(info, "now")
        self.assertEqual(row["config_json"], {"quantization_config": {"quant_method": "awq"}})
        self.assertEqual(row["card_data_json"]["tags"], ["awq"])
        self.assertEqual(row["card_data_json"]["library_name"], "mlx")
        self.assertEqual(row["card_data_json"]["quantization_config"], {"bits": 4})
        self.assertNotIn("model-index", row["card_data_json"])

    def test_completed_query_is_reused_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            query = {"filter": "gguf"}
            api = SimpleNamespace(list_models=lambda **kwargs: iter([self.info()]))
            first = collect.collect_query(api, query, cache)
            second = collect.collect_query(None, query, cache)
            self.assertEqual(first, second)
            self.assertEqual(len(first["rows"]), 1)

    def test_failed_query_does_not_leave_a_completed_cache(self):
        def fail(**kwargs):
            yield self.info()
            raise RuntimeError("interrupted pagination")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                collect.collect_query(SimpleNamespace(list_models=fail), {"search": "model"}, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_latest_observation_replaces_stale_metadata(self):
        old = {"observed_at": "2026-01-01", "query": {"repo_id": "pub/m"}, "rows": [{"id": "pub/m", "tags_json": ["old"]}]}
        new = {"observed_at": "2026-02-01", "query": {"filter": "gguf"}, "rows": [{"id": "Pub/M", "tags_json": ["gguf"]}]}
        self.assertEqual(collect.merge_rows([new, old]), new["rows"])

    def test_reject_incomplete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            query = {"filter": "gguf"}
            path = Path(directory) / f"{collect.query_key(query)}.json"
            path.write_text(json.dumps({"query": query, "complete": False}))
            with self.assertRaisesRegex(ValueError, "Invalid completed"):
                collect.collect_query(None, query, Path(directory))

    def test_failed_refresh_keeps_prior_completed_query_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            query = {"filter": "gguf"}
            api = SimpleNamespace(list_models=lambda **kwargs: iter([self.info()]))
            path = collect.cache_query(api, query, cache)
            original = path.read_bytes()
            def fail(**kwargs):
                yield self.info()
                raise RuntimeError("interrupted pagination")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                collect.cache_query(SimpleNamespace(list_models=fail), query, cache, refresh=True)
            self.assertEqual(path.read_bytes(), original)

    def test_streamed_collection_reuses_legacy_caches_and_matches_merge_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = [{"filter": "gguf"}, {"repo_id": "pub/a"}, {"filter": "mlx"}]
            results = [
                {"query": queries[0], "observed_at": "2026-01-01", "complete": True,
                 "rows": [{"id": "pub/a", "disabled": False}, {"id": "pub/b", "tags_json": ["gguf"]}]},
                {"query": queries[1], "observed_at": "2026-01-01", "complete": True,
                 "rows": [{"id": "Pub/A", "disabled": True}]},
                {"query": queries[2], "observed_at": "2026-02-01", "complete": True, "rows": []},
            ]
            for result in results:
                (root / f"{collect.query_key(result['query'])}.json").write_text(json.dumps(result))
            output = root / "out.jsonl"
            # The large CLI path must not fall back to loading any complete cache.
            with patch.object(Path, "read_text", side_effect=AssertionError("full cache read")):
                manifest = collect.collect_queries(None, queries, root, output)
            rows = list(map(json.loads, output.read_text().splitlines()))
            self.assertEqual(rows, collect.merge_rows(results))
            self.assertTrue(rows[0]["disabled"])
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["queries"][-1]["rows"], 0)
            original = output.read_bytes(), output.with_suffix(".jsonl.manifest.json").read_bytes()
            collect.collect_queries(None, queries, root, output)
            self.assertEqual(original, (output.read_bytes(), output.with_suffix(".jsonl.manifest.json").read_bytes()))

    def test_fresh_query_is_sorted_and_complete_without_building_a_rows_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.info(), self.info()
            a.id, b.id = "pub/a", "pub/b"
            api = SimpleNamespace(list_models=lambda **kwargs: iter([b, a]))
            output = root / "out.jsonl"
            result = collect.collect_queries(api, [{"filter": "gguf"}], root / "cache", output)
            self.assertEqual([row["id"] for row in map(json.loads, output.read_text().splitlines())], ["pub/a", "pub/b"])
            self.assertEqual(result["rows"], 2)
            self.assertEqual(len(list((root / "cache").iterdir())), 1)

    def test_private_or_truncated_cache_cannot_replace_prior_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query = {"filter": "gguf"}
            path = root / f"{collect.query_key(query)}.json"
            output = root / "out.jsonl"
            output.write_text("previous-output\n")
            for suffix, error in [('{"id":"secret/repo","private":true}]}', ValueError), ('{"id":"pub/a"}', ijson.JSONError)]:
                path.write_text(json.dumps({"query": query, "observed_at": "2026-01-01", "complete": True})[:-1] + ',"rows":[' + suffix)
                with self.assertRaises(error):
                    collect.collect_queries(None, [query], root, output)
                self.assertEqual(output.read_text(), "previous-output\n")

    def test_legacy_large_cache_merges_with_bounded_python_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query = {"filter": "gguf"}
            path = root / f"{collect.query_key(query)}.json"
            with path.open("w") as stream:
                stream.write(json.dumps({"query": query, "observed_at": "2026-01-01", "complete": True})[:-1] + ',"rows":[')
                for index in range(1000):
                    if index:
                        stream.write(",")
                    stream.write(json.dumps({"id": f"pub/m-{index:04}", "legacy_config": "x" * 16384}))
                stream.write("]}")
            self.assertGreater(path.stat().st_size, 16_000_000)
            tracemalloc.start()
            try:
                manifest = collect.collect_queries(None, [query], root, root / "out.jsonl")
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual(manifest["rows"], 1000)
            self.assertLess(peak, 3_000_000)


if __name__ == "__main__":
    unittest.main()
