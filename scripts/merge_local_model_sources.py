"""Merge normalized metadata inputs in priority order (last input wins).

Use a pinned snapshot first, then current public Hub observations. The manifest
records exact input and output hashes; this does not resolve parent ambiguities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_sources(inputs: list[Path], output: Path) -> dict:
    """Use a disk index so a broad Hub census does not require census-sized RAM."""
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = []
    with tempfile.TemporaryDirectory(prefix="local-model-merge-", dir=output.parent) as directory:
        with sqlite3.connect(str(Path(directory) / "metadata.sqlite")) as connection:
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("CREATE TABLE metadata(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            for path in inputs:
                count = replaced = 0
                seen = set()
                with path.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        model_id = row.get("id", "")
                        if not isinstance(model_id, str) or model_id.count("/") != 1:
                            raise ValueError(f"{path}:{line_number}: invalid model ID")
                        key = model_id.casefold()
                        if key in seen:
                            raise ValueError(f"{path}:{line_number}: duplicate ID {model_id}")
                        seen.add(key)
                        replaced += connection.execute("SELECT 1 FROM metadata WHERE id=?", (key,)).fetchone() is not None
                        payload = json.dumps(row, separators=(",", ":"), ensure_ascii=False)
                        connection.execute("INSERT INTO metadata VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (key, payload))
                        count += 1
                        if count % 5000 == 0:
                            connection.commit()
                connection.commit()
                source = {"name": path.name, "sha256": file_hash(path), "rows": count, "replaced": replaced}
                manifest_path = path.with_suffix(path.suffix + ".manifest.json")
                if manifest_path.exists():
                    collection = json.loads(manifest_path.read_text())
                    claimed_hash = collection.get("sha256", collection.get("output_sha256"))
                    if claimed_hash and claimed_hash != source["sha256"]:
                        raise ValueError(f"Collection manifest hash disagrees: {manifest_path}")
                    source["collection"] = collection
                sources.append(source)
            temporary = output.with_suffix(output.suffix + ".tmp")
            count = 0
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for (payload,) in connection.execute("SELECT payload FROM metadata ORDER BY id"):
                    stream.write(payload + "\n")
                    count += 1
            temporary.replace(output)
    manifest = {"schema_version": 1, "precedence": "last input wins by case-insensitive repository ID",
                "inputs": sources, "rows": count, "sha256": file_hash(output)}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in args.inputs}:
        parser.error("Output must be different from every input")
    manifest = merge_sources(args.inputs, args.output)
    print(f"Merged {manifest['rows']:,} repositories")


if __name__ == "__main__":
    main()
