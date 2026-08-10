# Tracked Models

The core HuggingFace models used to track metrics for [The ATOM Project](https://www.atomproject.ai/).

## Files

### `models.csv` (primary list)

The main tracked model list. These are the core frontier models used in ATOM Project charts and analysis.

### `extra_models.csv` (secondary list)

A secondary list of models that are tracked but not yet included in the main charts. These are candidates for promotion to `models.csv` in the future. Useful for broader ecosystem analysis, coverage of niche orgs, and filling gaps in existing org catalogs.

**New orgs in extra list**: AI-MO, AIDC-AI, apple, bigcode, CohereLabs, docling-project, GSAI-ML, h2oai, ibm-research, LGAI-EXAONE, LiquidAI, llm-jp, opendatalab, OpenHands, openvla, Salesforce, state-spaces, swiss-ai, TinyLlama, typhoon-ai

**Existing orgs with additional models**: allenai, arcee-ai, google, microsoft, Qwen

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

### 2026-08-10
- Weekly Hugging Face sync
- Added 3 models to `models.csv`: internlm, LiquidAI
- 23 candidates rejected, 7 left for manual review

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
