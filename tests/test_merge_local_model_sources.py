import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("merge_sources", Path(__file__).resolve().parents[1] / "scripts" / "merge_local_model_sources.py")
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


class MergeTests(unittest.TestCase):
    def test_fresh_source_overrides_casefold_identity_and_hashes_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, output = (root / name for name in ("old.jsonl", "new.jsonl", "out.jsonl"))
            old.write_text('{"id":"pub/a","tags_json":["old"]}\n{"id":"pub/b"}\n')
            new.write_text('{"id":"Pub/A","tags_json":["new"]}\n')
            report = merge.merge_sources([old, new], output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["id"] for r in rows], ["Pub/A", "pub/b"])
            self.assertEqual(rows[0]["tags_json"], ["new"])
            self.assertEqual(report["inputs"][1]["replaced"], 1)
            self.assertEqual(report["sha256"], merge.file_hash(output))

    def test_stale_manifest_cannot_claim_different_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "out.jsonl"
            source.write_text('{"id":"pub/a"}\n')
            source.with_suffix(".jsonl.manifest.json").write_text('{"sha256":"stale"}')
            with self.assertRaisesRegex(ValueError, "manifest hash disagrees"):
                merge.merge_sources([source], output)
            self.assertFalse(output.exists())

    def test_duplicate_identity_within_single_source_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "out"
            source.write_text('{"id":"pub/a"}\n{"id":"Pub/A"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate ID"):
                merge.merge_sources([source], output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
