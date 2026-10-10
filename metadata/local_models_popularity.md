# Popularity coverage review

This review checks whether the separate [local-model catalog](../local-models.csv)
misses obvious, widely downloaded repositories. It starts with global rankings,
without filtering to known local variants, publishers, or model tasks.

| Ranking | Observation | Repositories |
| --- | --- | ---: |
| Recent downloads | Public Hub API, October 10, 2026, 21:24 UTC; rolling-month `downloads` | 2,000 |
| Lifetime downloads | October 9, 2026 snapshot; top `downloads_all_time` across all 3,134,913 metric rows | 2,000 |

The two lists contain **2,829 unique repositories**. Their dates and ranking
universes differ; the recent list's lifetime counters do not make it a global
lifetime ranking. The pinned dataset revision is
`ef89c8f4e19c6b97ab5a74bfa6946517f8cde7c5`.

## Results and evidence

The audit adds **1,524 repositories**, bringing the catalog to **103,299**; none
were removed. It recognizes 1,381 AutoRound and 952 ModelOpt repositories,
including overlaps with existing format categories. Reviewed mappings also
repair descendants through intermediate checkpoints.

| Ranked sample | Catalog entries before → after | Newly covered download requests | Unresolved candidates after review |
| --- | ---: | ---: | ---: |
| Recent top 2,000 | 350 → 382 | 19,889,575 rolling-month | 109 |
| Lifetime top 2,000 | 170 → 191 | 171,005,215 lifetime | 47 |

The two download columns use different observation dates and must not be added
together. All 24 tracked parents in the recent global top 100 already had
cataloged variants, including supported conversion-only variants. This is a
coverage check, not a claim that every variant of those parents was found.

The review found two systematic omissions: **AutoRound** and **ModelOpt**
quantization evidence. Discovery now recognizes their supported tags and config
methods, including the `auto-round`, `autoround`, and `auto_round` spellings.
For example, [NVIDIA's Qwen3.6-35B-A3B NVFP4 release](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/blob/1355db6a052410cfd62085d94b58866fd0f2c3c5/README.md)
ranked 261st by lifetime downloads, while
[Lorbus's Qwen3.6-27B AutoRound release](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound/blob/c3aea2d531678621989e5e2db034e32b22536e79/README.md)
ranked 1,371st. A format-like name alone still does not establish format or parent.

Reviewed cards and configs also repair missing or misleading parent metadata.
Examples include the official MiniCPM5 GGUF/MLX releases, Qwen's
Qwen3.5-122B-A10B GPTQ release, kaitchup's Phi-3 GPTQ release, and
hugging-quants' Llama 3.1 AWQ releases. The MiniCPM5 DSpark variants retain the
separately tracked DSpark parent. The repository named `MiniCPM5-2B-GPTQ`
reports AWQ in its pinned config/tags; its name does not override that evidence.

These repairs preserve modification history. FireFunction and the reviewed
extended-context Llama descendants retain their fine-tuning lineage. Use
`is_modified == "false"` for supported conversion-only paths; modified and
uncertain paths remain distinguishable. See the [catalog contract](local_models.md).

The durable review records are:

- [Recent per-repository audit](local_models_popularity_recent.jsonl) and [summary](local_models_popularity_recent_report.json).
- [Lifetime per-repository audit](local_models_popularity_lifetime.jsonl) and [summary](local_models_popularity_lifetime_report.json).
- [Manual review](local_models_popularity_manual_review.json), including pinned evidence and reasons for fixes, exclusions, and unresolved cases.

The automated reports distinguish canonical repositories, catalog-covered local
repositories, unresolved local candidates, explicit task-scope exclusions, and
repositories with no supported local-format evidence in the supplied metadata.
They also give local-variant and conversion-only counts for popular canonical
parents. A low count is a review priority, not evidence that variants do not exist.
An unresolved count is not a count of confirmed missing eligible variants.

## Parent-cohort gaps and unresolved cases

Some popular local repositories have clear sources outside both canonical lists.
The reviewed examples include:

| Untracked source checkpoint | Example local repository |
| --- | --- |
| `Qwen/Qwen3-Coder-Next` | `unsloth/Qwen3-Coder-Next-GGUF` |
| `01-ai/Yi-Coder-1.5B-Chat`, `01-ai/Yi-Coder-9B-Chat`, `01-ai/Yi-1.5-6B-Chat` | Corresponding `MaziyarPanahi/*-GGUF` repositories |
| `upstage/solar-pro-preview-instruct` | `MaziyarPanahi/solar-pro-preview-instruct-GGUF` |
| `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | `rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF` |
| `baidu/Unlimited-OCR` | `sahilchachra/Unlimited-OCR-AWQ` |

These are parent-cohort gaps, with no reviewed path to another tracked ancestor.
Adding an original checkpoint requires a separate curated-list decision.
Ornith is a related but less certain case: its own originals are absent from the
cohort, and family-level Qwen/Gemma descriptions do not identify exact upstream
checkpoints. A model size or family resemblance cannot supply that lineage.

Other cases remain unresolved for different reasons. The SmolLM repository
`nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16` has conflicting artifacts: its
README describes the 135M model, while a config source path names 1.7B. The
LM Studio Gemma QAT variant names an unavailable exact parent; it must not be
silently reassigned to a similarly named tracked checkpoint. Mixed repositories
such as `ggml-org/models-moved` lack one attributable original. Embedding,
speech, image-generation, and older models can also have local-format files
without belonging to the current tracked LLM/VLM cohort.

This review changes neither `models.csv` nor `extra_models.csv`, and does not
change main dashboard totals. Popularity samples help prioritize review; they
are not an exhaustive census. HF downloads measure repository requests, not
unique users, and formats can use different counting rules.

## Refresh and reproduce

Run these commands from the repository root. To collect a fresh, unfiltered
**recent-download** sample using public metadata only:

```bash
uv run scripts/collect_model_popularity.py --top 2000 \
  --output .local-model-cache/popularity_recent.jsonl.gz

python scripts/audit_local_model_popularity.py --metric downloads --top 2000 \
  --metadata .local-model-cache/popularity_recent.jsonl.gz \
  --source-manifest .local-model-cache/popularity_recent.jsonl.gz.manifest.json \
  --candidates metadata/local_models_candidates.csv \
  --output .local-model-cache/popularity_recent_audit.jsonl \
  --report .local-model-cache/popularity_recent_report.json
```

A fresh Hub query is a new observation, not a replay of October 10. The collector
rejects incomplete, duplicate, out-of-order, or private results before publishing
an output. It fetches metadata and filenames, never model weights.

For a new **lifetime** sample, rank the complete chosen
`models_metrics/date=YYYY-MM-DD/part-0000.parquet` by `downloads_all_time DESC,
id ASC`, take the first 2,000, then join their metadata from the matching pinned
`models_dim.parquet`. Preserve the dataset revision, snapshot date, metric and
metadata file hashes, exact counters, and original rank in the input manifest.
Do not rank only the catalog, candidate cache, or the recent-download sample.
The checked-in lifetime input preserves the October 9 selection for offline replay.

Reproduce both checked-in audits without downloading the source snapshot or
calling the Hub API:

```bash
python scripts/audit_local_model_popularity.py --metric downloads --top 2000 \
  --metadata metadata/local_models_popularity_recent_input.jsonl.gz \
  --source-manifest metadata/local_models_popularity_recent_input.jsonl.gz.manifest.json \
  --candidates metadata/local_models_candidates.csv \
  --output metadata/local_models_popularity_recent.jsonl \
  --report metadata/local_models_popularity_recent_report.json

python scripts/audit_local_model_popularity.py --metric downloadsAllTime --top 2000 \
  --metadata metadata/local_models_popularity_lifetime_input.jsonl.gz \
  --source-manifest metadata/local_models_popularity_lifetime_input.jsonl.gz.manifest.json \
  --candidates metadata/local_models_candidates.csv \
  --output metadata/local_models_popularity_lifetime.jsonl \
  --report metadata/local_models_popularity_lifetime_report.json
```

Append `--check` to either command to compare the saved audit and report without
writing. CI runs both checks. The report pins the input, manifest, catalog,
canonical lists, candidates, and rule/script hashes. Source ranks remain separate
from deterministic ranks computed within the supplied input; missing metrics
are reported and excluded rather than replaced with zero.
