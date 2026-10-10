import csv
import json
from pathlib import Path
import sys

import pytest

duckdb = pytest.importorskip("duckdb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_local_model_snapshot import COLUMNS, extract_snapshot, normalize_metadata  # noqa: E402


def row(model_id, **values):
    return {**dict.fromkeys(COLUMNS), "id": model_id, "private": False, **values}


def write_parquet(path, rows):
    connection = duckdb.connect()
    types = {"private": "BOOLEAN", "disabled": "BOOLEAN"}
    connection.execute("CREATE TABLE metadata(" + ",".join(f"{key} {types.get(key, 'VARCHAR')}" for key in COLUMNS) + ")")
    connection.executemany("INSERT INTO metadata VALUES(" + ",".join("?" for _ in COLUMNS) + ")", [[value.get(key) for key in COLUMNS] for value in rows])
    connection.execute("COPY metadata TO ? (FORMAT PARQUET)", [str(path)])
    connection.close()


def write_parents(path, ids):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["org", "model", "modelId"])
        for model_id in ids:
            writer.writerow([*model_id.split("/"), model_id])


def test_full_candidate_selection_and_parent_closure_are_deterministic(tmp_path):
    source = tmp_path / "source.parquet"
    rows = [
        row("Qwen/Tracked-Root"), row("Other/Extra-Root"),
        row("u/tagged", tags_json='["gguf"]', base_model_ids_json='["u/intermediate"]'),
        row("u/intermediate", base_model_ids_json='["Qwen/Tracked-Root"]', card_data_json='{"base_model_relation":"quantized","base_model":"Qwen/Tracked-Root","irrelevant":"omit"}'),
        row("u/artifact-only", siblings_json='[{"rfilename":"export/model.onnx"}]'),
        row("u/opaque-packed", tags_json='["compressed-tensors"]'),
        row("u/opaque-library", library_name="compressed-tensors"),
        row("u/config-only", config_json='{"quantization_config":{"quant_method":"hqq"}}'),
        row("u/name-only-GGUF"), row("u/Tracked-Root-unmarked"),
        row("u/disabled", disabled=True, tags_json='["mlx"]'),
        row("u/private-GGUF", private=True, tags_json='["gguf"]'),
        row("u/unrelated"),
    ]
    write_parquet(source, rows)
    primary, extra = tmp_path / "models.csv", tmp_path / "extra.csv"
    write_parents(primary, ["Qwen/Tracked-Root"])
    write_parents(extra, ["Other/Extra-Root"])
    output, manifest = tmp_path / "metadata.jsonl", tmp_path / "manifest.json"
    result = extract_snapshot(source, primary, extra, output, manifest)
    models = {item["id"]: item for item in map(json.loads, output.read_text().splitlines())}
    assert set(models) == {item["id"] for item in rows} - {"u/private-GGUF", "u/unrelated"}
    assert models["u/disabled"]["disabled"] is True
    assert models["u/intermediate"]["card_data"] == {"base_model_relation": "quantized", "base_model": "Qwen/Tracked-Root"}
    assert models["u/artifact-only"]["siblings"] == ["export/model.onnx"]
    assert result["input_rows"] == 13
    assert result["public_input_rows"] == 12
    assert result["ancestor_rows"] == 1
    first = output.read_bytes(), manifest.read_bytes()
    extract_snapshot(source, primary, extra, output, manifest)
    assert first == (output.read_bytes(), manifest.read_bytes())


def test_private_metadata_is_rejected_and_explicit_card_relation_preserved():
    with pytest.raises(ValueError, match="Private"):
        normalize_metadata(row("private/repo", private=True))
    model = normalize_metadata(row("u/model", base_model_relation="quantized", card_data_json='{"base_model_relation":"finetune","base_model":"Qwen/root"}'))
    assert model["base_model_relation"] == "quantized"
    assert model["card_data"]["base_model_relation"] == "finetune"


def test_ancestor_closure_keeps_disabled_but_excludes_private_ancestors(tmp_path):
    source = tmp_path / "source.parquet"
    write_parquet(source, [
        row("Qwen/root"),
        row("u/visible", tags_json='["gguf"]', base_model_ids_json='["u/disabled-parent","u/private-parent","u/absent"]'),
        row("u/disabled-parent", disabled=True, card_data_json='{"base_model":"Qwen/root"}'),
        row("u/private-parent", private=True),
    ])
    primary, extra = tmp_path / "models.csv", tmp_path / "extra.csv"
    write_parents(primary, ["Qwen/root"])
    write_parents(extra, [])
    output = tmp_path / "metadata.jsonl"
    result = extract_snapshot(source, primary, extra, output, tmp_path / "manifest.json")
    models = {item["id"]: item for item in map(json.loads, output.read_text().splitlines())}
    assert set(models) == {"Qwen/root", "u/visible", "u/disabled-parent"}
    assert result["unavailable_or_private_ancestor_ids"] == 2


def test_duplicate_metrics_are_rejected_without_replacing_prior_output(tmp_path):
    source = tmp_path / "source.parquet"
    write_parquet(source, [row("Qwen/root")])
    primary, extra = tmp_path / "models.csv", tmp_path / "extra.csv"
    write_parents(primary, ["Qwen/root"])
    write_parents(extra, [])
    metrics = tmp_path / "metrics.parquet"
    connection = duckdb.connect()
    connection.execute("COPY (SELECT 'Qwen/root' AS id, 10 AS downloads, 20 AS downloads_all_time, 'sha' AS sha UNION ALL SELECT 'qwen/root', 11, 21, 'other') TO ? (FORMAT PARQUET)", [str(metrics)])
    connection.close()
    output = tmp_path / "metadata.jsonl"
    output.write_text("previous-output\n")
    with pytest.raises(ValueError, match="Duplicate metrics"):
        extract_snapshot(source, primary, extra, output, tmp_path / "manifest.json", metrics_parquet=metrics)
    assert output.read_text() == "previous-output\n"


def test_output_cannot_overwrite_source(tmp_path):
    source = tmp_path / "source.parquet"
    with pytest.raises(ValueError, match="distinct"):
        extract_snapshot(source, tmp_path / "models.csv", tmp_path / "extra.csv", source, tmp_path / "manifest.json")


def test_auto_round_and_modelopt_select_opaque_tags_libraries_and_name_hints(tmp_path):
    source = tmp_path / "source.parquet"
    aliases = ("auto-round", "autoround", "auto_round", "modelopt")
    rows = [row("Root/Base")]
    for index, alias in enumerate(aliases):
        rows.extend([
            row(f"u/tagged-{index}", tags_json=json.dumps([alias])),
            row(f"u/library-{index}", library_name=alias),
            row(f"u/checkpoint-{alias}"),
        ])
    rows.append(row("u/nonmodeloptimization"))
    write_parquet(source, rows)
    primary, extra = tmp_path / "models.csv", tmp_path / "extra.csv"
    write_parents(primary, ["Root/Base"])
    write_parents(extra, [])
    output = tmp_path / "metadata.jsonl"
    extract_snapshot(source, primary, extra, output, tmp_path / "manifest.json")
    actual = {json.loads(line)["id"] for line in output.read_text().splitlines()}
    assert actual == {item["id"] for item in rows[:-1]}
