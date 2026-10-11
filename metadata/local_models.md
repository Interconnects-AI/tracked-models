# Local-model catalog

[`local-models.csv`](../local-models.csv) curates major quantized and local-format
variants, including modified descendants, with supported lineage to a checkpoint
in `models.csv` or `extra_models.csv`. It does not change those lists or dashboard
totals. One row represents one repository, not a quantization file or user.

Include a variant when its publisher is the tracked parent's organization, or it
has at least **100,000 lifetime downloads** or **10,000 recent (30-day) downloads**.
The same-publisher rule prioritizes original providers, including new releases;
it does not label third-party quantizers as official. The catalog grows as tracked
parents publish variants or community variants meet the download thresholds.
There is no row-count cap. Broader discovery results stay local.
Thresholds use a pinned download snapshot; selection is not a complete census and
can favor established models. Keep the cohort fixed when comparing historical
months. The initial selection uses the October 9, 2026 snapshot.

## Schema and use

Rows are sorted case-insensitively by exact `modelId`, with Unix newlines.
The parent is the nearest tracked checkpoint in the recorded lineage.

| Column | Meaning |
| --- | --- |
| `org`, `model`, `modelId` | Variant publisher, repository name, and exact ID. |
| `parent_model_id`, `parent_org`, `parent_model` | Tracked checkpoint and its publisher/name. |
| `parent_region` | Parent publisher's `us`, `china`, `eu`, `other`, or `unknown` region, joined from `organization_regions.csv`. |
| `parent_list` | `models` or `extra_models`. |
| `local_formats` | Semicolon-separated, potentially overlapping format/quantization categories. |
| `base_model_id` | Immediate source; may be an untracked derivative or local variant. |
| `variant_category` | `conversion`, `finetune`, `pruned`, `adapter`, `merge`, `abliterated`, `distilled`, `mixed`, or `unknown`. |
| `is_modified` | `false` for supported conversion-only lineage; `true` for modified or uncertain paths. |
| `modification_categories` | Semicolon-separated changes after the tracked parent; blank for conversion-only paths. |
| `parent_lineage` | Semicolon-separated IDs from this repository to its tracked parent. |
| `lineage_categories` | One entry per lineage edge; simultaneous categories use `+`. |
| `mapping_method` | `direct`, `lineage_chain`, or `reviewed_override`. |
| `already_tracked` | Whether the repository is already in either canonical list. |
| `created_at` | Hub repository creation date, not a reviewed public release date. |

Filter `is_modified == "false"` to compare conversion-only adoption with the
broader catalog. Conversion-only includes official and community conversions.
A GGUF export of an untracked fine-tune remains modified relative
to its tracked ancestor. If the fine-tuned checkpoint is itself tracked, that
checkpoint is the baseline. `unknown` does not prove additional training occurred.

For supplemental primary-list totals, select `parent_list == "models"` and
exclude repositories already in the consumer's baseline. Count each repository
once across overlapping formats. Region describes the tracked parent, not the
variant publisher or users. HF downloads are requests, not unique users.

Supported formats are `gguf`, `mlx`, `awq`, `gptq`, `exl2`, `onnx`, `openvino`,
`hqq`, `exl3`, `bitsandbytes`, `aqlm`, `quanto`, `ggml`, `compressed-tensors`,
`auto-round`, and `modelopt`. Names alone never establish a format or parent.

## Refresh and validation

[Overrides](local_models_overrides.json) retain reviewed parent/category
corrections and exclusions; [aliases](local_models_aliases.json) retain verified
repository redirects. These small inputs prevent known metadata errors from
returning during refresh. They include supporting links and are loaded by default.
Metadata-based mappings are not independent verification of every publisher claim.

Collect public metadata and filenames, never weights. For routine refreshes,
put the existing catalog's lineage IDs and reviewed new candidates in an ID file
(one repository per line). Completed queries are cached; add `--refresh` for
fresh observations. Broad format/parent searches are optional local research,
not a required census on every refresh.

```bash
uv run scripts/collect_local_models.py \
  --repo-id-file .local-model-cache/reviewed-ids.txt \
  --cache .local-model-cache/hub --output .local-model-cache/hub.jsonl

# Optional pinned snapshot extraction; supply its actual dataset revision.
uv run --with duckdb scripts/extract_local_model_snapshot.py \
  --input-parquet /path/to/models_dim.parquet --dataset-revision REVISION \
  --output .local-model-cache/snapshot.jsonl \
  --manifest .local-model-cache/snapshot.jsonl.manifest.json

# Later inputs replace older observations of the same repository.
python scripts/merge_local_model_sources.py \
  .local-model-cache/snapshot.jsonl .local-model-cache/hub.jsonl \
  --output .local-model-cache/combined.jsonl

python scripts/local_models.py --metadata .local-model-cache/combined.jsonl \
  --source-manifest .local-model-cache/combined.jsonl.manifest.json \
  --popularity .local-model-cache/popularity.jsonl

python scripts/local_models.py --validate
python -m pytest tests -q
```

For Hub-only generation, use the Hub JSONL and its adjacent manifest directly.
Supply a pinned popularity JSONL (or Parquet) with one row per repository:
`id`, `downloads` (recent 30 days), and `downloads_all_time`. Nulls remain missing;
they do not qualify community repositories. Preserve its source/date locally.
The generation report records its hash and the selection counts. Do not commit
this download snapshot or the full discovery output.
Generation writes `local-models.csv`; evidence and reports stay under the ignored
`.local-model-cache/` directory. Add `--candidates .local-model-cache/review.csv`
for an unresolved review queue. Preserve cached inputs/manifests to replay the
same generation command with `--check`; fresh Hub observations can change results.

`--validate` checks the CSV contract, canonical parent/region joins, lineage,
modification flags, and reviewed corrections without raw source downloads or
committed evidence files. CI also runs the offline tests. Review generated CSV
diffs before publishing; missing or conflicting lineage remains unresolved, and
coverage is not an exhaustive census of local models.
