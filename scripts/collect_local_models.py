# /// script
# dependencies = ["huggingface_hub>=1,<2", "ijson>=3.3,<4"]
# ///
"""Collect public Hub metadata for the separate local-model catalog.

Search finds candidates only: scripts/local_models.py decides membership from
metadata and reviewed evidence. No model weights or private metadata are fetched.
Every completed query is cached so interrupted broad sweeps can be resumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CARD_FIELDS = ("base_model", "base_model_relation", "library_name", "pipeline_tag", "tags",
               "quantization_config", "quantization", "quantized_by")
FORMAT_FILTERS = (
    "gguf", "mlx", "awq", "gptq", "exl2", "exl3", "onnx", "openvino",
    "hqq", "bitsandbytes", "compressed-tensors", "gptqmodel", "aqlm", "quanto", "ggml",
)


def normalize_model(info: Any, observed_at: str) -> dict[str, Any] | None:
    """Keep public discovery facts, with immutable model revisions when supplied."""
    if getattr(info, "private", False):
        return None
    card = getattr(info, "card_data", None)
    card = card.to_dict() if hasattr(card, "to_dict") else card or {}
    bases = card.get("base_model") or []
    if isinstance(bases, str):
        bases = [bases]
    created = getattr(info, "created_at", None)
    return {
        "id": info.id,
        "library_name": getattr(info, "library_name", None),
        "tags_json": sorted(set(getattr(info, "tags", None) or [])),
        "base_model_relation": card.get("base_model_relation"),
        "base_model_ids_json": bases,
        "card_data_json": {key: card[key] for key in CARD_FIELDS if key in card},
        "private": False,
        "disabled": bool(getattr(info, "disabled", False)),
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        "config_json": {key: value for key, value in (getattr(info, "config", None) or {}).items()
                        if key == "quantization_config"},
        "siblings_json": sorted(s.rfilename for s in (getattr(info, "siblings", None) or [])),
        "sha": getattr(info, "sha", None),
        "source_url": f"https://huggingface.co/api/models/{info.id}",
        "observed_at": observed_at,
    }


def query_key(query: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:20]


def make_queries(args: argparse.Namespace) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    if args.all_formats:
        queries.extend({"filter": value} for value in FORMAT_FILTERS)
    queries.extend({"filter": value} for value in args.filter)
    queries.extend({"author": value} for value in args.author)
    queries.extend({"search": value} for value in args.search)
    queries.extend({"repo_id": value} for value in args.repo_id)
    if args.repo_id_file:
        queries.extend({"repo_id": line.strip()} for line in args.repo_id_file.read_text().splitlines()
                       if line.strip() and not line.startswith("#"))
    if args.parent_search or args.parent_links:
        parents: dict[str, dict[str, str]] = {}
        for path in args.parents:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    parents[row["modelId"]] = row
        for model_id, row in sorted(parents.items()):
            if args.parent_links:
                queries.append({"filter": f"base_model:{model_id}"})
            if args.parent_search:
                queries.append({"search": row["model"]})
    return list({query_key(q): q for q in queries}.values())


def cache_query(api: Any, query: dict[str, str], cache: Path, refresh: bool = False) -> Path:
    """Publish a completed, sorted query cache without keeping its rows in RAM."""
    path = cache / f"{query_key(query)}.json"
    if path.exists() and not refresh:
        return path
    observed = datetime.now(timezone.utc).isoformat()
    if "repo_id" in query:
        infos = [api.model_info(query["repo_id"])]
    else:
        infos = api.list_models(**query, full=True, cardData=True, fetch_config=True)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="query-", dir=cache) as directory:
        with sqlite3.connect(str(Path(directory) / "rows.sqlite")) as connection:
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("CREATE TABLE rows(id TEXT PRIMARY KEY, exact_id TEXT, payload TEXT)")
            for info in infos:
                row = normalize_model(info, observed)
                if row is not None:
                    payload = json.dumps(row, separators=(",", ":"), ensure_ascii=False)
                    try:
                        connection.execute("INSERT INTO rows VALUES(?,?,?)", (row["id"].casefold(), row["id"], payload))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Duplicate model identity in query: {row['id']}") from error
            connection.commit()
            temporary = Path(directory) / "completed.json"
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                header = {"query": query, "observed_at": observed, "complete": True}
                stream.write(json.dumps(header, separators=(",", ":"), ensure_ascii=False)[:-1] + ',"rows":[')
                separator = ""
                for (payload,) in connection.execute("SELECT payload FROM rows ORDER BY id,exact_id"):
                    stream.write(separator + payload)
                    separator = ","
                stream.write("]}\n")
            temporary.replace(path)
    return path


def query_metadata(path: Path, query: dict[str, str]) -> dict:
    """Read the small cache header without materializing its rows array."""
    import ijson

    header = {}
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream):
            if prefix == "rows" and event == "start_array":
                break
            # Header fields are deliberately written before rows in query caches.
            if prefix == "" and event == "start_map":
                header = {}
            elif prefix == "query" and event == "start_map":
                header["query"] = {}
            elif prefix.startswith("query.") and event == "string":
                header["query"][prefix.removeprefix("query.")] = value
            elif prefix in {"observed_at", "complete"} and event in {"string", "boolean"}:
                header[prefix] = value
        else:
            raise ValueError(f"Invalid completed query cache: {path}")
    if header.get("query") != query or header.get("complete") is not True or not isinstance(header.get("observed_at"), str):
        raise ValueError(f"Invalid completed query cache: {path}")
    return header


def collect_query(api: Any, query: dict[str, str], cache: Path, refresh: bool = False) -> dict:
    """Compatibility helper for small callers; the CLI uses streamed caches."""
    path = cache_query(api, query, cache, refresh)
    query_metadata(path, query)
    return json.loads(path.read_text(encoding="utf-8"))


def collect_queries(api: Any, queries: list[dict[str, str]], cache: Path, output: Path,
                    refresh: bool = False) -> dict:
    """Merge each cache into a disk index and stream the final JSONL artifact."""
    import ijson

    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with tempfile.TemporaryDirectory(prefix="local-model-collection-", dir=output.parent) as directory:
        with sqlite3.connect(str(Path(directory) / "metadata.sqlite")) as connection:
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("CREATE TABLE metadata(id TEXT PRIMARY KEY, observed TEXT, explicit INTEGER, payload TEXT)")
            for index, query in enumerate(queries, start=1):
                path = cache_query(api, query, cache, refresh)
                header = query_metadata(path, query)
                count = 0
                with path.open("rb") as stream:
                    for row in ijson.items(stream, "rows.item", use_float=True):
                        if not isinstance(row, dict) or row.get("private"):
                            raise ValueError(f"Private or invalid metadata in cache: {path}")
                        model_id = row.get("id")
                        if not isinstance(model_id, str) or model_id.count("/") != 1:
                            raise ValueError(f"Invalid model identity in cache: {path}")
                        payload = json.dumps(row, separators=(",", ":"), ensure_ascii=False)
                        connection.execute("""INSERT INTO metadata VALUES(?,?,?,?)
                            ON CONFLICT(id) DO UPDATE SET observed=excluded.observed,
                                explicit=excluded.explicit,payload=excluded.payload
                            WHERE excluded.observed > metadata.observed OR
                                (excluded.observed = metadata.observed AND excluded.explicit > metadata.explicit)""",
                            (model_id.casefold(), header["observed_at"], int("repo_id" in query), payload))
                        count += 1
                        if count % 5000 == 0:
                            connection.commit()
                connection.commit()
                summaries.append({"query": query, "observed_at": header["observed_at"], "rows": count})
                print(f"[{index}/{len(queries)}] {query}: {count} public models", file=sys.stderr, flush=True)
            temporary = Path(directory) / "metadata.jsonl"
            digest = hashlib.sha256()
            count = 0
            with temporary.open("wb") as stream:
                for (payload,) in connection.execute("SELECT payload FROM metadata ORDER BY id"):
                    line = (payload + "\n").encode("utf-8")
                    stream.write(line)
                    digest.update(line)
                    count += 1
            manifest = {
                "schema_version": 1, "source": "public Hugging Face Hub API", "complete": True,
                "queries": summaries, "rows": count, "sha256": digest.hexdigest(),
                "scope": "Complete query pagination, not an exhaustive census of all local variants. Names are discovery only.",
            }
            manifest_temporary = Path(directory) / "manifest.json"
            manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output)
            manifest_temporary.replace(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def merge_rows(results: list[dict]) -> list[dict]:
    """Most recent observation wins; explicit model_info wins timestamp ties."""
    rows: dict[str, tuple[tuple[str, bool], dict]] = {}
    for result in results:
        priority = (result["observed_at"], "repo_id" in result["query"])
        for row in result["rows"]:
            key = row["id"].casefold()
            if key not in rows or priority > rows[key][0]:
                rows[key] = (priority, row)
    return [rows[key][1] for key in sorted(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-formats", action="store_true", help="Exhaust every supported format-tag listing (can be large)")
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--author", action="append", default=[])
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--repo-id", action="append", default=[])
    parser.add_argument("--repo-id-file", type=Path)
    parser.add_argument("--parent-search", action="store_true", help="Search every tracked checkpoint name, including untagged variants")
    parser.add_argument("--parent-links", action="store_true", help="Query every tracked base-model tag")
    parser.add_argument("--parents", type=Path, nargs="+", default=[ROOT / "models.csv", ROOT / "extra_models.csv"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="Replace completed query caches with current Hub observations")
    args = parser.parse_args()
    queries = make_queries(args)
    if not queries:
        parser.error("Choose --all-formats, --parent-search, --parent-links, or explicit queries")
    from huggingface_hub import HfApi
    # This registry is public. Deliberately limit discovery to anonymous public metadata.
    api = HfApi(token=False)
    manifest = collect_queries(api, queries, args.cache, args.output, args.refresh)
    print(f"Wrote {manifest['rows']:,} metadata rows to {args.output}")


if __name__ == "__main__":
    main()
