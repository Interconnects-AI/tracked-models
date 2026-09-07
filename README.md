# Tracked Models

The core HuggingFace models used to track metrics for [The ATOM Project](https://www.atomproject.ai/).

## Files

### `models.csv` (primary list)

The main tracked model list. These are the core frontier models used in ATOM Project charts and analysis.

### `extra_models.csv` (secondary list)

A secondary list of models that are tracked but not yet included in the main charts. These are candidates for promotion to `models.csv` in the future. Useful for broader ecosystem analysis, coverage of niche orgs, and filling gaps in existing org catalogs.

**New orgs in extra list**: AI-MO, AIDC-AI, apple, bigcode, CohereLabs, docling-project, GSAI-ML, h2oai, ibm-research, LGAI-EXAONE, LiquidAI, llm-jp, marin-community, opendatalab, OpenHands, openvla, Salesforce, state-spaces, swiss-ai, TinyLlama, typhoon-ai

**Existing orgs with additional models**: allenai, arcee-ai, google, microsoft, Qwen

### `arxiv_models.csv` (research-paper mentions)

A separate family-level list for tracking model mentions in arXiv papers. Unlike
`models.csv` and `extra_models.csv`, its rows describe model families rather
than downloadable checkpoints. Inclusion here does not promote a model into the
download-tracking lists.

The list contains 42 curated historical families and five additional families
tracked only in new daily scans: Muse Glimmer, Inkling, Laguna, Apertus, and
Marin. The scanner's frozen historical taxonomy still contains all 51 raw
families; this CSV is an intentionally curated view, not a one-row-per-family
copy of that source taxonomy. Future-only families are not backfilled into old
papers and do not appear in the featured public chart unless separately
promoted. Apertus is published by Swiss AI; Marin is published by the Marin
community in collaboration with Open Athena. See
[the original request](https://github.com/Interconnects-AI/tracked-models/issues/38).

The `olmo` row uses the preferred **OLMo** display label as an umbrella for the
AI2-published OLMo, Tulu, and Molmo model families. LLaVA, Vicuna, Alpaca,
OpenLLaMA, RWKV, LLaDA, and CogVLM/CogVideo are intentionally omitted from
this curated list, but remain intact in the frozen raw taxonomy and historical
source data. TinyLlama is not attributed to Meta's Llama family. Tencent-
qualified Hy2 and Hy3 names are illustrative Hunyuan-lineage aliases.

| Column | Description |
|--------|-------------|
| `family_id` | Stable family identifier shared with the arXiv scanner or future-only watchlist |
| `label` | Human-readable model-family name |
| `hf_org` | Official Hugging Face provider namespace when available; otherwise blank |
| `access_class` | `open_weight_family` or `proprietary` |
| `tracking_scope` | `historical_and_daily` for a curated historical family, or `future_only` for new-paper monitoring |
| `aliases` | Semicolon-separated illustrative names, not the executable matching rules |

Historical family definitions come from the scanner's
[canonical taxonomy](https://github.com/Interconnects-AI/arxiv-model-mentions/blob/main/arxiv_mentions_lib/arxiv_model_families.json).
Prospective matching rules are maintained separately in the dashboard's
[future-only arXiv watchlist](https://github.com/Interconnects-AI/dashboard/blob/main/pipeline/arxiv_emerging_families.json).
Provider-qualified and model-specific aliases prevent ambiguous ordinary words
from being counted as research-paper model mentions.

The `hf_org` column identifies the primary family publisher. Historical aliases
describe the curated presentation intent, including the OLMo umbrella and
prospective Hunyuan names; they are not guaranteed to mirror every executable
alias in the frozen scanner. Editing this informational CSV does not change
live matching, raw taxonomy membership, or previously calculated charts;
changing those requires separately reviewed scanner or dashboard updates.

### `metadata/`

Curated, project-neutral model metadata for reuse across analysis projects:

- `model_parameters.csv`: canonical total/active parameter counts, MoE classification, and concise notes.
- `model_sizes.py`: generated legacy-compatible total-parameter map and analysis helpers.
- `generate_model_sizes.py`: metadata validator and compatibility-map generator.
- `organization_regions.csv`: canonical ATOM region buckets for exact Hugging Face organization namespaces, including explicit `unknown` classifications.
- `release_date_corrections.csv`: public release-date corrections for models whose Hugging Face `created_at` differs from the actual release date.

Legacy values remain usable and are labeled `Legacy data; not recently
reviewed.` in notes. Supporting links belong in the correction issue or pull
request history rather than the CSV. See
[`metadata/README.md`](metadata/README.md) for the contract and correction
workflow.

## Format

The primary and secondary model-list CSV files share the same three columns:

| Column | Description |
|--------|-------------|
| `org` | HuggingFace organization/user |
| `model` | Model name |
| `modelId` | Full model identifier (`org/model`) |

## Usage

```bash
# Fetch the primary list
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/models.csv

# Fetch the extra list
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/extra_models.csv

# Fetch the arXiv model-family list
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/arxiv_models.csv

# Combine both lists (skip extra header)
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/models.csv > all_models.csv
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/extra_models.csv | tail -n +2 >> all_models.csv

# Get just the model IDs from primary list
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/models.csv | tail -n +2 | cut -d',' -f3
```

## Scope

Post-ChatGPT LLMs and VLMs (released after Nov 30, 2022) with first-party weights on HuggingFace. Threshold: >100K total downloads for new additions (exceptions for notable recent releases).

### Original orgs

The project began with private daily download data from HuggingFace covering 7 organizations (1,971 models through July 10, 2025):

`deepseek-ai`, `google`, `meta-llama`, `microsoft`, `mistral-community`, `mistralai`, `Qwen`

The tracked list has since expanded to cover additional frontier model providers, VLM families, and historically significant LLM orgs.

## Excluded Models

Models intentionally excluded from tracking despite high download counts:

- **Guard/shield models** (Llama-Guard, ShieldGemma, Qwen3Guard, granite-guardian, wildguard, gpt-oss-safeguard, etc.) -- safety classifiers, not generative LLMs.

## Changelog

### 2026-09-07
- Weekly Hugging Face sync
- Added 8 models to `models.csv`: deepseek-ai, inclusionAI, Qwen, tencent
- 33 candidates rejected, 4 left for manual review

### 2026-09-03
- Classified EleutherAI and BigCode as US using institutional attribution, while retaining their international research context
- Documented TII's UAE attribution without changing its region bucket

### 2026-08-31
- Weekly Hugging Face sync
- Added 8 models to `models.csv`: Qwen, tencent, zai-org
- 25 candidates rejected, 8 left for manual review

### 2026-08-25
- Added `arxiv_models.csv` as a separate family-level research-paper tracking list
- Documented 42 curated historical families and five future-only families, including Apertus and Marin
- Grouped Tulu and Molmo under OLMo, removed low-priority historical families from the curated view, and clarified Llama and Hunyuan aliases
- Added three official Marin checkpoints and eight additional Apertus checkpoints to `extra_models.csv`
- Classified `marin-community` as a US organization in the canonical region registry

### 2026-08-24
- Weekly Hugging Face sync
- Added 9 models to `models.csv`: inclusionAI, LiquidAI
- 28 candidates rejected, 7 left for manual review

### 2026-08-20
- Added Moondream 3.1 and LongCat 2.0 to the primary list with family discovery rules
- Added reviewed parameter metadata for 12 Artifacts Log issue 23 checkpoints plus two release-date corrections; the other 10 remain unranked RAM-only metadata
- Added 5 current Artifacts Log checkpoints to `models.csv`: CohereLabs, dots-studio, meta-models, Motif-Technologies, nvidia
- Promoted North Micro Vision from `extra_models.csv` and added reviewed parameter metadata for all 5 checkpoints plus Ling 3.0 tiny
- Added four official release-date corrections and discovery coverage for the new primary families

### 2026-08-19
- Added 5 headline checkpoints to `models.csv`: CohereLabs, moonshotai, thinkingmachines, Zyphra
- Promoted Command A+ from `extra_models.csv` and added reviewed Kimi K3, Inkling, Inkling-Small, and ZAYA1 parameter metadata

### 2026-08-17
- Weekly Hugging Face sync
- Added 14 models to `models.csv`: deepseek-ai, inclusionAI, internlm, LiquidAI, nvidia, Qwen
- Added 6 models to `extra_models.csv`: CohereLabs, nvidia
- 21 candidates rejected, 1 left for manual review

### 2026-08-10
- Weekly Hugging Face sync
- Added 21 models to `models.csv`: inclusionAI, internlm, LiquidAI, poolside
- Added 2 models to `extra_models.csv`: LGAI-EXAONE
- 24 candidates rejected, 0 left for manual review

### 2026-08-03
- Weekly Hugging Face sync
- Added 3 models to `models.csv`: deepseek-ai, LiquidAI
- Added 3 models to `extra_models.csv`: LGAI-EXAONE, swiss-ai
- 15 candidates rejected, 4 left for manual review

### 2026-07-20
- Weekly Hugging Face sync
- Added 2 models to `models.csv`: internlm
- 32 candidates rejected, 6 left for manual review

### 2026-07-13
- Weekly Hugging Face sync
- Added 2 models to `extra_models.csv`: nvidia
- 33 candidates rejected, 11 left for manual review

### 2026-07-06
- Weekly Hugging Face sync
- Added 6 models to `models.csv`: deepseek-ai, LiquidAI, Qwen, tencent
- 37 candidates rejected, 9 left for manual review

### 2026-06-15
- Weekly Hugging Face sync
- Added 5 models to `models.csv`: google, MiniMaxAI, moonshotai, XiaomiMiMo
- Added 2 models to `extra_models.csv`: CohereLabs
- 26 candidates rejected, 5 left for manual review

### 2026-06-08
- Weekly Hugging Face sync
- Added 18 models to `models.csv`: google, nvidia
- Added 3 models to `extra_models.csv`: LiquidAI
- 47 candidates rejected, 1 left for manual review

### 2026-06-01
- Weekly Hugging Face sync
- Added 2 models to `extra_models.csv`: LiquidAI
- 33 candidates rejected, 3 left for manual review

### 2026-05-25
- Weekly Hugging Face sync
- Added 10 models to `models.csv`: openbmb
- Added 5 models to `extra_models.csv`: AIDC-AI, CohereLabs, opendatalab
- 42 candidates rejected, 0 left for manual review

### 2026-05-17
- Weekly Hugging Face sync
- Added 67 models to `models.csv` across 18 orgs: allenai, arcee-ai, ByteDance-Seed, deepseek-ai, google, HuggingFaceTB, ibm-granite, inclusionAI, internlm, MiniMaxAI, mistralai, moonshotai, nvidia, openbmb, Qwen, tencent, XiaomiMiMo, zai-org
- Added 18 models to `extra_models.csv` across 7 orgs: AIDC-AI, apple, LGAI-EXAONE, LiquidAI, llm-jp, nvidia, opendatalab
- 262 candidates rejected, 0 left for manual review

### 2026-03-26
- Synced new HuggingFace models since the last CSV update with `sync_hf_models.py`
- Added new models to `models.csv` across 15 existing orgs: allenai, arcee-ai, baidu, ibm-granite, inclusionAI, internlm, microsoft, MiniMaxAI, mistralai, nvidia, openbmb, Qwen, rednote-hilab, tencent, zai-org
- Added new models to `extra_models.csv` across 8 orgs: AIDC-AI, CohereLabs, GSAI-ML, LGAI-EXAONE, LiquidAI, llm-jp, opendatalab, OpenHands

### 2026-02-08
- Added `extra_models.csv` with 119 models across 25 orgs (20 new, 5 existing)
- Secondary tracking list for broader ecosystem coverage, can be promoted to `models.csv`

### 2026-02-07
- Added 169 models across 9 new orgs and 11 existing orgs
- Removed 11 guard/shield models (granite-guardian, Qwen3Guard, vaultgemma)
- New orgs: OpenGVLab, tiiuae, baichuan-inc, llava-hf, EleutherAI, facebook, moondream, vikhyatk, rednote-hilab
- Filled gaps in google, microsoft, nvidia, meta-llama, ByteDance-Seed, and other existing orgs
- Scope: post-ChatGPT models only (released after Nov 30, 2022) with >100K total downloads

### 2026-01-09
- Added arcee-ai Trinity collection (8 models): Trinity-Mini, Trinity-Mini-Base, Trinity-Mini-Base-Pre-Anneal, Trinity-Mini-GGUF, Trinity-Nano-Base, Trinity-Nano-Base-Pre-Anneal, Trinity-Nano-Preview, Trinity-Nano-Preview-GGUF
- Added arcee-ai AFM 4.5B series (5 models): AFM-4.5B, AFM-4.5B-Base, AFM-4.5B-GGUF, AFM-4.5B-ov, AFM-4.5B-Preview

## License

Apache 2.0
