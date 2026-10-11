# Local-model catalogs

[`local-models.csv`](../local-models.csv) tracks major quantized and local-format
**conversions of checkpoints in `models.csv`**. The separate
[`local-models-extra.csv`](../local-models-extra.csv) is a backup for conversions
of `extra_models.csv` checkpoints; default consumers exclude it. Neither file
changes the canonical lists or main chart totals. One row is one repository,
not a quantization file or user.

Every edge to the tracked parent must be a supported conversion. Fine-tuned,
pruned, adapted, merged, abliterated, distilled, and uncertain descendants are
excluded, even when popular or published by the parent organization. A tracked
instruct/fine-tuned checkpoint is itself the baseline; a direct conversion of it
is eligible. A conversion of an untracked fine-tune of that checkpoint is not.

New conversions qualify when published by the parent organization, or with at
least **100,000 lifetime downloads** or **10,000 recent (30-day) downloads**.
Previously admitted conversions stay when downloads decline or are missing,
provided refreshed metadata still establishes eligible conversion-only lineage.
There is no row-count cap. The initial admission snapshot is October 9, 2026.
This is a curated cohort, not a complete census; keep membership fixed when
comparing historical months. Broad discovery and audit data stay outside Git.

## Schema and use

Both files use the same schema, sorted case-insensitively by exact `modelId`,
with Unix newlines. The parent is the nearest tracked checkpoint in the lineage.

| Column | Meaning |
| --- | --- |
| `org`, `model`, `modelId` | Variant publisher, repository name, and exact ID. |
| `parent_model_id`, `parent_org`, `parent_model` | Tracked checkpoint and publisher/name. |
| `parent_region` | Parent publisher's region from `organization_regions.csv`. |
| `parent_list` | Always `models` in the default file; `extra_models` in the backup. |
| `local_formats` | Semicolon-separated, potentially overlapping format categories. |
| `base_model_id` | Immediate source; may be an intermediate conversion. |
| `variant_category`, `is_modified` | Always `conversion` and `false` in public catalogs. |
| `modification_categories` | Empty in public catalogs. |
| `parent_lineage` | Semicolon-separated IDs from this repository to its tracked parent. |
| `lineage_categories` | One `conversion` entry per lineage edge. |
| `mapping_method` | `direct`, `lineage_chain`, or `reviewed_override`. |
| `already_tracked` | Whether the repository is already in either canonical list. |
| `created_at` | Hub creation date, not a reviewed public release date. |

For supplemental totals, exclude repositories already in the consumer's baseline.
Count each repository once across overlapping formats. Region describes the
tracked parent, not the variant publisher or users. Downloads are requests, not
unique users.

Supported formats include GGUF, MLX, AWQ, GPTQ, EXL2/3, ONNX, OpenVINO, HQQ,
bitsandbytes, AQLM, Quanto, GGML, compressed-tensors, AutoRound, and ModelOpt.
NVFP4 and similar precision variants remain eligible when configuration, tags,
or artifacts support a recognized conversion format. Names alone establish
neither format nor parent.

## Refresh and validation

[Overrides](local_models_overrides.json) hold reviewed parent/category corrections
and exclusions; [aliases](local_models_aliases.json) hold verified parent redirects.
They include evidence links and load by default. Metadata-based mappings do not
independently verify every publisher claim.

Collect public metadata and filenames, never weights. For routine updates, collect
lineage IDs from **both existing catalogs**, include-override repositories, and
reviewed new candidates. Completed
queries are cached; use `--refresh` for new observations. Broader searches and
snapshot extraction remain optional local research.

```bash
uv run scripts/collect_local_models.py \
  --repo-id-file .local-model-cache/reviewed-ids.txt \
  --cache .local-model-cache/hub --output .local-model-cache/hub.jsonl

python scripts/merge_local_model_sources.py \
  .local-model-cache/snapshot.jsonl .local-model-cache/hub.jsonl \
  --output .local-model-cache/combined.jsonl

python scripts/local_models.py --metadata .local-model-cache/combined.jsonl \
  --source-manifest .local-model-cache/combined.jsonl.manifest.json \
  --popularity .local-model-cache/popularity.jsonl

python scripts/local_models.py --validate
python -m pytest tests -q
```

For Hub-only generation, pass its JSONL and adjacent manifest directly. Supply a
pinned popularity JSONL or Parquet with `id`, `downloads` (recent 30 days), and
`downloads_all_time`. Nulls cannot admit a new community repository. Keep source
snapshots, dates, manifests, and reports locally; the report records input hashes.

Generation reads both current CSVs as admission history, freshly validates source
lineage, and writes both outputs. `--output` and `--extra-output` override their
paths. Preserve the published CSV pair with pinned inputs for reproducible replay;
`--check` compares generated artifacts. Evidence and reports default to ignored
`.local-model-cache/`; use `--candidates` there for an optional unresolved queue.

A canonical parent promotion/demotion moves its conversions between the files
without resetting admission. A previously admitted conversion with missing or
newly ineligible lineage stops generation, rather than silently disappearing.
Refresh missing metadata, or record a reviewed exclusion/category correction with
evidence before removal. If a parent leaves both canonical lists, review its
lineage and disposition explicitly. Do not use declining popularity as a removal
reason. Explain corrections, parent moves, and removals in the PR.

`--validate` checks both CSVs, canonical parent/region joins, conversion-only
lineage, disjoint membership, and reviewed corrections without downloading raw
sources. Full generation also validates source evidence. CI runs both CSV
validation and offline tests; review the resulting diffs before publication.
