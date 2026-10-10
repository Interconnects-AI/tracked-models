import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.collect_model_popularity import collect


def info(model_id, downloads, **kwargs):
    return SimpleNamespace(id=model_id, downloads=downloads, **kwargs)


class FakeApi:
    def __init__(self, rows):
        self.rows = rows
        self.query = None

    def list_models(self, **query):
        self.query = query
        yield from self.rows


def test_unfiltered_rank_and_counters_survive_compression(tmp_path):
    api = FakeApi([info("z/model", 100, downloads_all_time=500), info("a/model", 20)])
    output = tmp_path / "ranking.jsonl.gz"
    manifest = collect(api, output, 2)
    rows = [
        json.loads(line) for line in gzip.decompress(output.read_bytes()).splitlines()
    ]
    assert [(row["id"], row["audit_rank"]) for row in rows] == [
        ("a/model", 2),
        ("z/model", 1),
    ]
    assert rows[1]["downloads_all_time"] == 500
    assert api.query["sort"] == "downloads"
    assert api.query["token"] is False
    assert set(api.query) == {"sort", "limit", "expand", "token"}
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.with_suffix(".gz.manifest.json").read_text()) == manifest


@pytest.mark.parametrize(
    "rows,message",
    [
        ([info("a/model", 10)], "Incomplete ranking"),
        ([info("a/model", 10), info("A/Model", 5)], "Duplicate repository"),
        ([info("a/model", 10), info("b/model", 15)], "not ordered"),
        ([info("a/model", 10), info("b/model", None)], "Invalid recent"),
        ([info("a/model", 10), info("b/model", 1, private=True)], "private repository"),
    ],
)
def test_invalid_or_partial_response_preserves_previous_artifact(
    tmp_path, rows, message
):
    output = tmp_path / "ranking.jsonl"
    output.write_text("previous artifact\n")
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text("previous manifest\n")
    with pytest.raises(ValueError, match=message):
        collect(FakeApi(rows), output, 2)
    assert output.read_text() == "previous artifact\n"
    assert manifest.read_text() == "previous manifest\n"


@pytest.mark.parametrize("lifetime", [-1, True, 1.5, "100"])
def test_invalid_lifetime_counter_preserves_existing_output(tmp_path, lifetime):
    output = tmp_path / "ranked.jsonl"
    output.write_text("previous\n")
    with pytest.raises(ValueError, match="Invalid lifetime"):
        collect(FakeApi([info("a/model", 1, downloads_all_time=lifetime)]), output, 1)
    assert output.read_text() == "previous\n"


def test_interrupted_pagination_does_not_publish_partial_ranking(tmp_path):
    class InterruptedApi:
        def list_models(self, **query):
            assert query["token"] is False
            yield info("a/model", 10)
            raise RuntimeError("network interruption")

    output = tmp_path / "ranked.jsonl.gz"
    output.write_bytes(b"previous")
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text("previous manifest")
    with pytest.raises(RuntimeError, match="interruption"):
        collect(InterruptedApi(), output, 2)
    assert output.read_bytes() == b"previous"
    assert manifest.read_text() == "previous manifest"


def test_overlong_response_is_rejected_and_ties_keep_source_order(tmp_path):
    output = tmp_path / "ranked.jsonl"
    with pytest.raises(ValueError, match="Overlong ranking"):
        collect(FakeApi([info("b/model", 1), info("a/model", 1)]), output, 1)
    assert not output.exists()
    collect(FakeApi([info("b/model", 1), info("a/model", 1)]), output, 2)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["id"], row["audit_rank"]) for row in rows] == [
        ("a/model", 2),
        ("b/model", 1),
    ]


def test_local_format_sweep_includes_new_quantization_aliases():
    from argparse import Namespace
    from scripts.collect_local_models import make_queries

    args = Namespace(
        all_formats=True,
        filter=[],
        author=[],
        search=[],
        repo_id=[],
        repo_id_file=None,
        parent_search=False,
        parent_links=False,
    )
    queries = make_queries(args)
    assert {"auto-round", "autoround", "auto_round", "modelopt"}.issubset(
        {query["filter"] for query in queries}
    )
