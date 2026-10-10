# Local-model catalog

`../local-models.csv` is a separate catalog of downloadable local-format and
quantized variants of checkpoints in `models.csv` and `extra_models.csv`,
including fine-tuned and pruned descendants with supported parent lineage.
It does not add repositories to those lists or change any dashboard totals.
Consumers must opt in to this catalog. It counts repositories, not individual
quantization files, model users, or installations.

The parent identifies the nearest tracked checkpoint in the recorded lineage.
For example, a conversion of a tracked DeepSeek distilled checkpoint belongs to
that DeepSeek checkpoint, not to the earlier model used to train it. The parent
region is joined from `organization_regions.csv`; it is an attribution of the
original tracked model, not the location of the quantization publisher or users.

## Row contract

One row per exact Hugging Face repository ID, sorted case-insensitively by
`modelId`, with Unix newlines. The first three columns match the existing model
lists for straightforward reuse.

| Column | Meaning |
| --- | --- |
| `org`, `model`, `modelId` | Publisher, repository name, and exact variant repository ID. |
| `parent_model_id`, `parent_org`, `parent_model` | Original tracked checkpoint and its publisher/name. |
| `parent_region` | Parent publisher's canonical `us`, `china`, `eu`, `other`, or `unknown` region. |
| `parent_list` | `models` for the primary list or `extra_models` for the secondary list. |
| `local_formats` | Semicolon-separated format or quantization categories; one repository may support several. |
| `base_model_id` | Immediate source, which may itself be a local variant or an untracked fine-tune. |
| `variant_category` | `conversion`, `finetune`, `pruned`, `adapter`, `merge`, `abliterated`, `distilled`, `mixed`, or `unknown`, relative to the tracked parent. |
| `is_modified` | `false` for a supported conversion-only lineage; `true` for modified or uncertain descendants. |
| `modification_categories` | Semicolon-separated changes found anywhere after the tracked parent; blank for conversion-only paths. |
| `parent_lineage` | Semicolon-separated repository IDs, starting with this variant and ending at its tracked parent. |
| `lineage_categories` | One semicolon-separated entry per edge in `parent_lineage`; simultaneous edge categories use `+`. |
| `mapping_method` | How the parent mapping was established. |
| `already_tracked` | Whether this repository is already present in either tracked list. |
| `created_at` | Hub repository creation date, when available; **not** a reviewed public release date. |

Join exact parent IDs to `model_parameters.csv` for curated parameter metadata.
Do not infer sizes from variant filenames or copy parameters across related
checkpoints.

A consumer adding supplemental counts to existing primary-list totals must
select `parent_list == "models"` and exclude repositories already in its own
baseline cohort. Never sum one copy of a repository for every `local_formats`
value; those categories overlap. Repository download counts also cannot be
interpreted as unique users.

For a comparison without fine-tuned, pruned, or uncertain descendants, filter
`is_modified == "false"`. Keep all rows for the broader comparison. The flag
propagates through intermediate repositories: a plain GGUF conversion of an
untracked fine-tune is still modified relative to its tracked ancestor. A
conversion of a fine-tuned checkpoint that is itself tracked uses that checkpoint
as its baseline and does not inherit changes from before that baseline.
`unknown` is deliberately excluded by the conversion-only filter; it does not
assert that additional training definitely occurred.

For example, `quant/Model-GGUF;tuner/Model;origin/Model` with
`lineage_categories=conversion;finetune` is `variant_category=finetune` and
`is_modified=true`. The category summarizes the path, while the edge list
preserves which repository introduced each change. `mixed` means more than one
modification category, including any uncertainty, occurs along the path.
An `uncensored` label alone is insufficient evidence of a specific training or
abliteration technique and is treated as uncertain.

```python
import csv

with open("local-models.csv", newline="") as stream:
    variants = list(csv.DictReader(stream))
conversion_only = [row for row in variants if row["is_modified"] == "false"]
# Compare a download-data join on exact modelId for variants vs conversion_only.
```

## Evidence and review

The generated evidence file records format signals, immediate-parent evidence,
parent paths, relationship categories, and whether the relationship was explicitly declared in a
model card or inferred/reported by the Hub. A base-model link alone can mean a
fine-tune, merge, adapter, or quantization. Discovery therefore checks format
signals and relationship evidence, traverses supported single-parent lineage,
and routes ambiguous mappings to review. The inherited region describes the
tracked ancestor; it does not describe the authors of a later fine-tune.

Name searches broaden candidate discovery; a similar name alone never assigns
a parent. Reviewed include/exclude exceptions live in
`local_models_overrides.json`, with immutable model-card links and explanations.
These exceptions can correct misleading Hub tags and classify additional
fine-tuning, pruning, or abliteration. Reviewed parent aliases coalesce verified
Hub redirects without guessing equivalent checkpoints from similar names.
The alias registry records the observed redirect and resolved repository revision.
Automated metadata mappings are evidence-based candidates, not a claim that
all publisher statements have been independently verified.

The report and candidate audit make gaps visible. Not every unresolved model
is in scope: many have several parents, lack a usable path to a tracked parent,
or publish unrelated non-LLM models. Coverage describes the observed
source and declared query scopes, not an exhaustive census of every local
model. Missing tags, renamed repositories, deleted repositories, and absent or
incorrect lineage remain limitations.

## Initial coverage

The October 10, 2026 catalog combines 802,310 public metadata observations
selected from a complete 3,137,236-row pinned metadata snapshot and targeted
public Hub queries. It includes 103,299 repositories from 21,008 publishers,
covering 1,160 of 1,840 primary parents and 133 of 199 secondary parents.
There are 65,026 GGUF repositories among the 16 overlapping format categories.

| Lineage classification | Repositories |
| --- | ---: |
| Conversion only (`is_modified=false`) | 23,640 |
| Fine-tuned | 30,340 |
| Pruned | 472 |
| Adapter | 7,369 |
| Merge with a supported single-parent declaration | 297 |
| Abliterated | 1,443 |
| Distilled | 179 |
| Mixed modifications or uncertainty | 17,522 |
| Unknown | 22,037 |

All categories except conversion-only have `is_modified=true`, totaling 79,659
repositories. Five catalog repositories already occur in a tracked list; use
`already_tracked` and the consumer's exact baseline cohort to prevent duplicate
counting. There are 62,439 associated entries in the review queue. The report's
larger unresolved count also includes unrelated or unsupported candidates and
must not be interpreted as 289,212 confirmed missing local variants.

Per-repository evidence is stored as deterministic gzip JSON Lines in
`local_models_evidence.jsonl.gz`; decompress with Python's standard `gzip`
module or `gzip -dc`. The CSV remains uncompressed. The validator checks the
compressed artifact hash and validates every decompressed record against its
CSV row, parent, reviewed mappings, and lineage categories.

The [popularity review](local_models_popularity.md) independently checks global
recent and lifetime download leaders and records repaired mappings, missing
format categories, and unresolved parent-cohort gaps.

## Refresh and validation

Public Hub collection fetches metadata and filenames, never model weights. It
exhausts each requested query's pagination, caches completed queries, and stops
on errors instead of writing a partial output as a completed sweep. Cached
queries support resuming interrupted runs; use `--refresh` for a fresh census.
Queries and cross-query merging use temporary SQLite indexes and streamed
JSON, so broad sweeps do not accumulate the full set of responses in memory.
Public disabled repositories are retained as exclusion observations so a newer
query can retire an older snapshot entry; private metadata is rejected.

```bash
# Install only the collection dependencies in an isolated environment, or use uv.
uv run scripts/collect_local_models.py \
  --all-formats --parent-search --parent-links \
  --cache .local-model-cache/hub \
  --output .local-model-cache/hub.jsonl

# For a targeted coverage audit, use --search, --author, or --repo-id.
uv run scripts/collect_local_models.py \
  --search Qwen3-8B --author OpenVINO \
  --cache .local-model-cache/hub \
  --output .local-model-cache/audit.jsonl
```

The complete format/name sweep can be large. Broad candidate collection is
separate from deterministic membership decisions, so refreshes can be reviewed
before committing a changed catalog. There is no scheduled publisher or change
to the existing tracked-model sync job.

To reproduce discovery from an existing `models_dim.parquet` snapshot:

```bash
uv run --with duckdb scripts/extract_local_model_snapshot.py \
  --input-parquet /path/to/models_dim.parquet \
  --dataset-revision ef89c8f4e19c6b97ab5a74bfa6946517f8cde7c5 \
  --output .local-model-cache/snapshot.jsonl \
  --manifest .local-model-cache/snapshot.jsonl.manifest.json

# Later inputs replace older metadata for the same repository.
python scripts/merge_local_model_sources.py \
  .local-model-cache/snapshot.jsonl .local-model-cache/hub.jsonl \
  --output .local-model-cache/combined.jsonl

cp .local-model-cache/combined.jsonl.manifest.json metadata/local_models_sources.json

python scripts/local_models.py \
  --metadata .local-model-cache/combined.jsonl \
  --source-manifest metadata/local_models_sources.json \
  --source-url https://github.com/Interconnects-AI/tracked-models/blob/codex/local-model-catalog/metadata/local_models_sources.json \
  --candidates metadata/local_models_candidates.csv

# Validate checked-in artifacts without a network connection or raw snapshot.
python scripts/local_models.py --validate \
  --source-manifest metadata/local_models_sources.json \
  --candidates metadata/local_models_candidates.csv

# Repeat with the exact same metadata input to verify byte-for-byte generation.
python scripts/local_models.py --check \
  --metadata .local-model-cache/combined.jsonl \
  --source-manifest metadata/local_models_sources.json \
  --source-url https://github.com/Interconnects-AI/tracked-models/blob/codex/local-model-catalog/metadata/local_models_sources.json \
  --candidates metadata/local_models_candidates.csv
```

The checked-in source manifest records the initial snapshot revision, the
specific current Hub queries, and all input/output hashes. The snapshot is from
`interconnects/hf-dumps` and requires dataset access to download; the public Hub
collector provides an alternative refresh path. Raw multi-gigabyte inputs stay
outside Git. Current Hub searches are observations made at different times,
not an atomic historical Hub snapshot. Exact replay requires the recorded
cached inputs and the same generation options; a fresh Hub sweep will change
as repositories and cards change.

`local_models_candidates.csv` is a compact review queue for unresolved or
excluded candidates associated with a tracked parent through metadata, lineage,
or name. It also includes tracked originals for explicit audit, which must not
be treated as missing conversions. A name association is only a search hint.
The report counts all considered candidates, including out-of-scope and
unassociated repositories not written to this queue. For a full external audit,
pass `--all-candidates --candidates .local-model-cache/all-candidates.jsonl`.
JSONL retains detailed evidence; the compact CSV carries IDs, format categories,
reasons, and lineage. Neither queue is a list approved for download aggregation.

Supported format categories are `gguf`, `mlx`, `awq`, `gptq`, `exl2`, `onnx`,
`openvino`, `hqq`, `exl3`, `bitsandbytes`, `aqlm`, `quanto`, `compressed-tensors`,
`auto-round`, `modelopt`, and legacy `ggml`.
These mix serialization/runtime ecosystems and quantization methods, so they
are deliberately not mutually exclusive. Filename evidence is required when
format tags/configuration do not identify a category. Format names occurring
only in repository names remain review hints. Other ecosystems and methods are
outside this initial taxonomy and can be added with evidence and tests.
