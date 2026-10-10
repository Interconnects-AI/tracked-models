import csv
import gzip
import hashlib
import json

import pytest

from scripts.audit_local_model_popularity import audit, main, select_ranked


def write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path, rows):
    payload = "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    path.write_bytes(
        gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload
    )


def setup_inputs(tmp_path, rows, catalog_rows=()):
    metadata, catalog = tmp_path / "ranked.jsonl", tmp_path / "catalog.csv"
    models, extra = tmp_path / "models.csv", tmp_path / "extra.csv"
    write_metadata(metadata, rows)
    columns = ["org", "model", "modelId"]
    write_csv(
        models, columns, [{"org": "Root", "model": "Base", "modelId": "Root/Base"}]
    )
    write_csv(
        extra, columns, [{"org": "Extra", "model": "Chat", "modelId": "Extra/Chat"}]
    )
    write_csv(
        catalog,
        [
            "modelId",
            "parent_model_id",
            "local_formats",
            "variant_category",
            "is_modified",
        ],
        list(catalog_rows),
    )
    return metadata, catalog, models, extra


def local_entry(model_id, category="conversion", parent="Root/Base"):
    return {
        "modelId": model_id,
        "parent_model_id": parent,
        "local_formats": "gguf",
        "variant_category": category,
        "is_modified": str(category != "conversion").lower(),
    }


def test_ranks_are_separate_metrics_with_alias_missing_and_deterministic_ties(tmp_path):
    rows = [
        {"id": "z/Recent", "downloads": 100, "downloads_all_time": 110, "rank": 41},
        {"id": "a/Tie", "downloads": 100, "downloadsAllTime": 200, "rank": 40},
        {"id": "b/Lifetime", "downloads": 1, "downloadsAllTime": 300, "rank": 42},
        {"id": "c/NoMetrics"},
    ]
    paths = setup_inputs(tmp_path, rows)
    result, report = audit(*paths, top=1)
    by_id = {row["model_id"]: row for row in result}
    assert set(by_id) == {"a/Tie", "b/Lifetime"}
    assert by_id["a/Tie"]["ranks"] == {"downloads": 1}
    assert by_id["a/Tie"]["source_rank"] == 40
    assert by_id["b/Lifetime"]["ranks"] == {"downloadsAllTime": 1}
    assert report["rankings"]["downloads"]["missing_metric_repositories"] == 1
    assert report["rankings"]["downloadsAllTime"]["cutoff_downloads"] == 300
    write_metadata(paths[0], list(reversed(rows)))
    replay, _ = audit(*paths, top=1)
    assert result == replay


def test_coverage_distinguishes_formats_names_tasks_and_popular_parent_variants(
    tmp_path,
):
    rows = [
        {"id": "root/base", "tags": ["gguf"]},
        {"id": "Extra/Chat"},
        {"id": "local/covered", "tags": ["gguf"]},
        {
            "id": "Local/Files",
            "siblings": ["weights/model.gguf"],
            "base_model_ids": ["Root/Base"],
        },
        {"id": "Local/Name-GGUF"},
        {
            "id": "Local/Encoder",
            "pipeline_tag": "feature-extraction",
            "siblings": ["onnx/model.onnx"],
        },
        {
            "id": "Local/MaybeChat",
            "pipeline_tag": "feature-extraction",
            "tags": ["gguf", "text-generation"],
        },
        {
            "id": "Local/ConflictingTask",
            "pipeline_tag": "feature-extraction",
            "tags": ["gguf"],
            "base_model_ids": ["Root/Base"],
        },
        {"id": "Other/Ordinary", "pipeline_tag": "text-generation"},
        {"id": "Local/Disabled", "disabled": True, "tags": ["gguf"]},
    ]
    for index, row in enumerate(rows):
        row["downloads"] = 100 - index
    paths = setup_inputs(
        tmp_path,
        rows,
        [
            local_entry("Local/Covered"),
            local_entry("Local/Modified", "finetune"),
            local_entry("Root/Base"),
        ],
    )
    candidates = tmp_path / "candidates.csv"
    write_csv(
        candidates,
        ["model_id", "reason", "local_formats", "base_model_ids", "lineage"],
        [
            {
                "model_id": "Local/Files",
                "reason": "missing_parent_evidence",
                "local_formats": "gguf",
                "base_model_ids": "Root/Base",
                "lineage": "Local/Files;Root/Base",
            }
        ],
    )
    result, report = audit(*paths, metrics=("downloads",), candidates=candidates)
    by_id = {row["model_id"]: row for row in result}
    parent = by_id["root/base"]
    assert parent["disposition"] == "canonical_main"
    assert parent["catalog_entry"] is not None
    assert parent["parent_catalog_coverage"] == {
        "local_variant_repos": 3,
        "conversion_repos": 2,
        "modified_repos": 1,
    }
    assert by_id["Extra/Chat"]["parent_catalog_coverage"]["local_variant_repos"] == 0
    assert by_id["Extra/Chat"]["canonical_list"] == "extra_models"
    assert by_id["local/covered"]["disposition"] == "catalog_covered_local"
    files = by_id["Local/Files"]
    assert files["disposition"] == "unresolved_local_candidate"
    assert files["reason"] == "missing_parent_evidence"
    assert files["review_priority"] == "supported_format"
    assert files["format_evidence"][0]["kind"] == "file"
    assert files["declared_tracked_parents"] == ["Root/Base"]
    name = by_id["Local/Name-GGUF"]
    assert name["review_priority"] == "unconfirmed_hint"
    assert name["local_formats"] == []
    assert by_id["Local/Encoder"]["disposition"] == "outside_model_scope"
    assert by_id["Local/MaybeChat"]["disposition"] == "unresolved_local_candidate"
    assert by_id["Local/ConflictingTask"]["disposition"] == "unresolved_local_candidate"
    assert by_id["Other/Ordinary"]["disposition"] == "no_local_format_evidence"
    assert by_id["Local/Disabled"]["disposition"] == "unavailable"
    assert report["rankings"]["downloads"]["unresolved_supported_format"] == 3
    assert (
        report["rankings"]["downloads"]["unresolved_name_or_prior_candidate_only"] == 1
    )


def test_reuses_encoded_core_config_and_explicit_base_evidence(tmp_path):
    paths = setup_inputs(
        tmp_path,
        [
            {
                "id": "local/opaque",
                "downloads": 9,
                "config_json": json.dumps(
                    {
                        "quantization_config": {"quant_method": "compressed-tensors"},
                        "chat_template": "omit me",
                    }
                ),
                "card_data_json": {
                    "base_model": "Root/Base",
                    "base_model_relation": "quantized",
                },
                "sha": "pinnedrevision",
                "observed_at": "2026-10-10T00:00:00Z",
            }
        ],
    )
    rows, _ = audit(*paths, metrics=("downloads",))
    assert rows[0]["local_formats"] == ["compressed-tensors"]
    assert rows[0]["base_evidence"]["relation_provenance"] == "publisher_card"
    assert rows[0]["sha"] == "pinnedrevision"
    assert "omit me" not in json.dumps(rows)


@pytest.mark.parametrize("value", [-1, True, 1.5, "5"])
def test_bad_metric_values_fail(tmp_path, value):
    paths = setup_inputs(tmp_path, [{"id": "x/model", "downloads": value}])
    with pytest.raises(ValueError, match="invalid downloads"):
        audit(*paths, metrics=("downloads",))


def test_conflicting_alias_duplicate_private_and_missing_metric_fail(tmp_path):
    cases = [
        (
            [{"id": "x/model", "downloadsAllTime": 5, "downloads_all_time": 6}],
            "Conflicting",
        ),
        (
            [{"id": "x/model", "downloads": 1}, {"id": "X/Model", "downloads": 2}],
            "Duplicate",
        ),
        ([{"id": "x/model", "downloads": 1, "private": True}], "Private"),
        ([{"id": "x/model", "downloads": None}], "No nonmissing"),
    ]
    for rows, error in cases:
        paths = setup_inputs(tmp_path, rows)
        with pytest.raises(ValueError, match=error):
            audit(*paths, metrics=("downloads",))


def test_gzip_input_and_source_manifest_hash_check(tmp_path):
    paths = list(setup_inputs(tmp_path, [{"id": "x/model", "downloads": 1}]))
    compressed = tmp_path / "ranked.jsonl.gz"
    write_metadata(compressed, [{"id": "x/model", "downloads": 1}])
    paths[0] = compressed
    manifest = tmp_path / "source.json"
    provenance = {
        "sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
        "sort": "downloads",
        "limit": 2000,
        "observed_at": "2026-10-10",
    }
    manifest.write_text(json.dumps(provenance))
    rows, report = audit(*paths, metrics=("downloads",), source_manifest=manifest)
    assert len(rows) == 1
    assert report["source"] == provenance
    manifest.write_text('{"sha256":"wrong"}')
    with pytest.raises(ValueError, match="manifest hash"):
        audit(*paths, metrics=("downloads",), source_manifest=manifest)


def test_cli_replays_and_does_not_overwrite_inputs_or_failed_outputs(tmp_path):
    paths = setup_inputs(tmp_path, [{"id": "x/model", "downloads": 1}])
    output, report = tmp_path / "audit.jsonl", tmp_path / "report.json"
    args = [
        "--metadata",
        str(paths[0]),
        "--catalog",
        str(paths[1]),
        "--models",
        str(paths[2]),
        "--extra",
        str(paths[3]),
        "--metric",
        "downloads",
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    main(args)
    original = output.read_bytes(), report.read_bytes()
    main(args + ["--check"])
    assert original == (output.read_bytes(), report.read_bytes())
    output.write_text("changed")
    with pytest.raises(ValueError, match="audit differs"):
        main(args + ["--check"])
    assert output.read_text() == "changed"
    with pytest.raises(SystemExit):
        main(args + ["--output", str(paths[0])])
    write_metadata(
        paths[0], [{"id": "private/secret", "downloads": 2, "private": True}]
    )
    with pytest.raises(ValueError, match="Private"):
        main(args)
    assert output.read_text() == "changed"
    assert report.read_bytes() == original[1]


def test_selection_retains_only_cutoff_rows_and_checks_positive_cutoff(tmp_path):
    path = tmp_path / "many.jsonl"
    write_metadata(path, [{"id": f"x/model{i}", "downloads": i} for i in range(3000)])
    selected, report = select_ranked(path, ("downloads",), 2)
    assert set(selected) == {"x/model2998", "x/model2999"}
    assert report["downloads"]["rankable_input_repositories"] == 3000
    paths = setup_inputs(tmp_path, [{"id": "x/model", "downloads": 1}])
    with pytest.raises(ValueError, match="positive cutoff"):
        audit(*paths, metrics=("downloads",), top=0)


def test_metric_specific_source_ranks_and_manifest_row_count(tmp_path):
    paths = setup_inputs(
        tmp_path,
        [
            {
                "id": "x/model",
                "downloads": 7,
                "downloads_all_time": 50,
                "audit_rank": 17,
                "lifetime_rank": 93,
            }
        ],
    )
    rows, _ = audit(*paths)
    assert rows[0]["source_rank"] == 17
    assert rows[0]["source_ranks"] == {"downloads": 17, "downloadsAllTime": 93}
    assert rows[0]["ranks"] == {"downloads": 1, "downloadsAllTime": 1}
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"rows":2}')
    with pytest.raises(ValueError, match="row count"):
        audit(*paths, source_manifest=manifest)
    write_metadata(
        paths[0],
        [{"id": "x/model", "downloads": 7, "audit_rank": 17, "lifetime_rank": True}],
    )
    with pytest.raises(ValueError, match="source rank"):
        audit(*paths)
