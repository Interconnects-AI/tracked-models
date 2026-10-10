# /// script
# dependencies = ["huggingface_hub>=1,<2"]
# ///
"""Collect a global public Hub ranking for the local-model coverage audit.

The Hub's downloads sort is recent (rolling-month) activity. Lifetime counters
are retained for reference, but this sample is not a global lifetime ranking.
Only metadata is fetched; there is no task, publisher, or format filter.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .collect_local_models import normalize_model
except ImportError:
    from collect_local_models import normalize_model


EXPAND = [
    "cardData",
    "config",
    "createdAt",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "library_name",
    "pipeline_tag",
    "private",
    "sha",
    "siblings",
    "tags",
]


def collect(api, output: Path, top: int = 2000) -> dict:
    if type(top) is not int or top < 1:
        raise ValueError("top must be positive")
    observed = datetime.now(timezone.utc).isoformat()
    query = {"sort": "downloads", "limit": top, "expand": EXPAND}
    rows, seen = [], set()
    previous = None
    for rank, info in enumerate(api.list_models(**query, token=False), 1):
        if rank > top:
            raise ValueError(f"Overlong ranking: expected {top} rows")
        row = normalize_model(info, observed)
        if row is None:
            raise ValueError("Unexpected private repository in public ranking")
        key = row["id"].casefold()
        if key in seen:
            raise ValueError(f"Duplicate repository in ranking: {row['id']}")
        seen.add(key)
        downloads = getattr(info, "downloads", None)
        if type(downloads) is not int or downloads < 0:
            raise ValueError(f"Invalid recent download counter: {row['id']}")
        if previous is not None and downloads > previous:
            raise ValueError("Hub response is not ordered by descending downloads")
        previous = downloads
        lifetime = getattr(info, "downloads_all_time", None)
        if lifetime is not None and (type(lifetime) is not int or lifetime < 0):
            raise ValueError(f"Invalid lifetime download counter: {row['id']}")
        row.update(
            downloads=downloads,
            downloads_all_time=lifetime,
            pipeline_tag=getattr(info, "pipeline_tag", None),
            audit_rank=rank,
        )
        rows.append(row)
    if len(rows) != top:
        raise ValueError(
            f"Incomplete ranking: expected {top} rows, received {len(rows)}"
        )
    raw = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in sorted(rows, key=lambda row: row["id"].casefold())
    )
    payload = raw
    if output.suffix == ".gz":
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
            stream.write(raw)
        payload = buffer.getvalue()
    manifest = {
        "schema_version": 1,
        "source": "public Hugging Face Hub API",
        "query": query,
        "token": False,
        "observed_at": observed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "scope": f"First {top} repositories sorted by Hub recent downloads; no pipeline or format filter. "
        "Recent downloads reflect the Hub rolling month, not total historical downloads.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "ranking"
        temporary.write_bytes(payload)
        temporary_manifest = Path(directory) / "manifest"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
        temporary_manifest.replace(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi

    manifest = collect(HfApi(token=False), args.output, args.top)
    print(f"Wrote {manifest['rows']:,} public ranking observations to {args.output}")


if __name__ == "__main__":
    main()
