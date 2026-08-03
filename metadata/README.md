# Curated Metadata

Neutral metadata for tracked Hugging Face models that is useful across
Interconnects analysis projects.

## Files

### `model_parameters.csv`

Canonical, referenceable total and active parameter counts for exact Hugging
Face checkpoints.

| Column | Description |
|--------|-------------|
| `model_id` | Exact, case-sensitive Hugging Face `org/checkpoint` ID. |
| `total_params_b` | Official architectural/reported total in decimal billions. |
| `active_params_b` | Explicit per-token active count in decimal billions, when stated by a source. |
| `count_status` | `verified`, `estimated`, or `needs_source`. |
| `source_url` | Official model card, paper, or launch-document URL. Required for `verified` and `estimated`. |
| `notes` | Provenance, rounding, architecture, or migration context. |

Total parameters are not checkpoint file size or a raw safetensors tensor sum.
Active parameters are never inferred from total parameters, including for dense
models. Store one explicit row per checkpoint; do not inherit values across a
family or normalize model ID casing.

Only source-backed `verified` and `estimated` rows are eligible for public Hub
sorting. `needs_source` preserves useful legacy knowledge without publishing it
as authoritative metadata. An `estimated` value is still source-backed: it is
used when the source itself reports an approximation.

### `model_sizes.py`

Compatibility helpers used by `Interconnects-AI/open-model-analysis`.

The module includes:

- `MANUAL_SIZES`: a generated legacy-compatible model ID to total-parameter map.
- `get_size_bucket(size_b)`: shared size-bucket labels.
- `parse_size_from_name(model_id)`: fallback parsing for `7B`, `72B`, etc.
- `get_model_size(model_id, safetensors_params=None)`: resolution order used by
  downstream analysis.
- `resolve_model_sizes(df, params_col='safetensors_parameters_json')`: pandas
  helper for adding `size_b` and `size_bucket`.

`MANUAL_SIZES` intentionally preserves the old helper API, including unsourced
legacy values, and therefore is not a publishable verification source. Rows
whose notes begin with `Legacy MANUAL_SIZES entry.` form that compatibility
set. New canonical rows do not automatically expand the legacy override map.

After editing `model_parameters.csv`, regenerate or check the literal map:

```bash
python metadata/generate_model_sizes.py --write
python metadata/generate_model_sizes.py --check
```

The checker validates exact headers, unique IDs, positive decimal values,
allowed statuses, source URLs, `active_params_b <= total_params_b`, and generated
map parity. CI runs the same check.

### `release_date_corrections.csv`

Manual release-date corrections for models where Hugging Face `created_at`
does not match the public release date.

| Column | Description |
|--------|-------------|
| `model_id` | Full Hugging Face model identifier. |
| `hf_created_at` | Hugging Face repository creation date. |
| `actual_release_date` | Curated public release date to use for release-relative analysis. |
| `notes` | Reason or source context for the correction. |

## Conventions

- Dates use `YYYY-MM-DD`.
- Parameter counts are stored as billions to match common model naming and
  analysis conventions.
- Total and active parameters remain separate for MoE models.
- Metadata here is intentionally public and project-neutral; analysis-specific
  code should consume these files rather than duplicating local corrections.

## Source Reliability

Hugging Face API metadata is useful for discovery and fallback only. Repository
`created_at` can precede a public launch because weights are often uploaded
privately first. Raw safetensors parameter counts can include auxiliary modules
such as vision towers, projectors, or multi-token-prediction heads. Model names
can advertise rounded values, active parameters, or a family size instead of an
exact architectural total. Promote metadata only when an official model card,
paper, or launch document explicitly supports the value.

To report an incorrect release date or parameter count, use the
[model metadata correction form](https://github.com/Interconnects-AI/tracked-models/issues/new?template=model-metadata-correction.yml).

## Hub Notification

Changes to `model_parameters.csv` or `release_date_corrections.csv` on `main`
dispatch `tracked-models-metadata-updated` to `Interconnects-AI/artifacts-hub`.
Configure the `HUB_DISPATCH_TOKEN` Actions secret with permission to create a
repository dispatch in the Hub repository. The Hub's scheduled sync remains the
fallback if a dispatch is missed.

## Downstream Contract

Downstream projects depend on this directory's file names, field names, and
basic data shapes. Do not change the format of `model_parameters.csv`,
`model_sizes.py`, or `release_date_corrections.csv` without coordinating matching
updates in those consumers.
