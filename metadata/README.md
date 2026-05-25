# Curated Metadata

Neutral metadata for tracked Hugging Face models that is useful across
Interconnects analysis projects.

## Files

### `model_sizes.py`

Manual model-size metadata and helper functions copied from
`Interconnects-AI/open-model-analysis`.

The module includes:

- `MANUAL_SIZES`: curated model ID to total parameter count in billions.
- `get_size_bucket(size_b)`: shared size-bucket labels.
- `parse_size_from_name(model_id)`: fallback parsing for `7B`, `72B`, etc.
- `get_model_size(model_id, safetensors_params=None)`: resolution order used by
  downstream analysis.
- `resolve_model_sizes(df, params_col='safetensors_parameters_json')`: pandas
  helper for adding `size_b` and `size_bucket`.

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
- For MoE models, `model_sizes.py` follows the source convention of total
  parameters rather than active parameters.
- Metadata here is intentionally public and project-neutral; analysis-specific
  code should consume these files rather than duplicating local corrections.

## Downstream Contract

Downstream projects depend on this directory's file names, field names, and
basic data shapes. Do not change the format of `model_sizes.py` or
`release_date_corrections.csv` without coordinating matching updates in those
consumers.
