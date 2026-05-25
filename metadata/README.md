# Curated Metadata

Neutral metadata for tracked Hugging Face models that is useful across
Interconnects analysis projects.

## Files

### `model_parameter_counts.csv`

Manual parameter-count metadata for models whose size is missing from
Hugging Face metadata, ambiguous for MoE architectures, or not reliably
parseable from the model name.

| Column | Description |
|--------|-------------|
| `model_id` | Full Hugging Face model identifier. |
| `total_params_b` | Total parameter count in billions. For MoE models, this is total parameters rather than active parameters. |
| `group` | Optional family or source-context note carried over from the curation source. |
| `notes` | Optional row-specific caveat, estimate marker, or additional context. |

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
- Metadata here is intentionally public and project-neutral; analysis-specific
  code should consume these files rather than duplicating local corrections.
