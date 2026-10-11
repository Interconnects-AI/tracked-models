import unittest
import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

from metadata.audit_parameter_coverage import coverage, main, tracked_ids

HEADER = "model_id,total_params_b,active_params_b,is_moe,notes\n"


class ParameterCoverageTests(unittest.TestCase):
    def test_primary_only_and_strict_total_threshold(self):
        report = coverage({"a/dense", "a/moe", "a/missing"},
                          {"a/dense": 100, "a/moe": 101, "extra/large": 500})
        self.assertEqual(report["primary_models"], 3)
        self.assertEqual(report["known_sizes"], 2)
        self.assertEqual(report["over_100b"], ["a/moe"])
        self.assertEqual(report["at_100b_boundary"], ["a/dense"])
        self.assertEqual(report["missing_sizes"], ["a/missing"])

    def test_baseline_does_not_hide_new_model_or_lost_metadata(self):
        report = coverage({"a/old_gap", "a/new", "a/lost"}, {}, {"a/old_gap"})
        self.assertEqual(report["newly_missing_sizes"], ["a/lost", "a/new"])
        self.assertEqual(len(report["missing_sizes"]), 3)

    def test_filled_gaps_are_not_carried_forward(self):
        report = coverage({"a/resolved"}, {"a/resolved": 7}, {"a/resolved", "a/retired"})
        self.assertEqual(report["missing_sizes"], [])
        self.assertEqual(report["newly_missing_sizes"], [])

    def test_ids_are_exact_and_rows_must_join(self):
        self.assertEqual(tracked_ids("org,model,modelId\nQwen,Root,Qwen/Root\n"), {"Qwen/Root"})
        for text in (
            "org,model,modelId\nQwen,Root,qwen/Root\n",
            "org,model,modelId\nQwen,Root,Qwen/Root\nQwen,Root,Qwen/Root\n",
            "org,model,id\nQwen,Root,Qwen/Root\n",
            "org,model,modelId\nQwen,Bad Name,Qwen/Bad Name\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tracked_ids(text)

    def test_git_baseline_flags_new_gap_and_metadata_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            models = root / "models.csv"
            parameters = root / "metadata/model_parameters.csv"
            output = root / "coverage.json"
            models.write_text("org,model,modelId\na,old,a/old\na,known,a/known\n")
            parameters.write_text(HEADER + "a/known,180,,false,\n")
            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True,
                                      capture_output=True, text=True)
            git("init", "-q")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                "commit", "-qm", "Baseline")
            args = ["--models", str(models), "--parameters", str(parameters),
                    "--repo", str(root), "--json-output", str(output), "--check"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
                self.assertEqual(main(args + ["--base-ref", "HEAD"]), 0)
                models.write_text(models.read_text() + "a,new,a/new\n")
                parameters.write_text(HEADER + "extra/only,7,,false,\n")
                self.assertEqual(main(args + ["--base-ref", "HEAD"]), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["newly_missing_sizes"], ["a/known", "a/new"])
            self.assertEqual(report["missing_sizes"], ["a/known", "a/new", "a/old"])


if __name__ == "__main__":
    unittest.main()
