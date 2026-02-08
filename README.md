# Tracked Models

The core HuggingFace models used to track metrics for [The ATOM Project](https://www.atomproject.ai/).

## Files

### `models.csv` (primary list)

The main tracked model list. These are the core frontier models used in ATOM Project charts and analysis.

### `extra_models.csv` (secondary list)

A secondary list of models that are tracked but not yet included in the main charts. These are candidates for promotion to `models.csv` in the future. Useful for broader ecosystem analysis, coverage of niche orgs, and filling gaps in existing org catalogs.

**New orgs in extra list**: AI-MO, AIDC-AI, apple, bigcode, CohereLabs, docling-project, GSAI-ML, h2oai, ibm-research, LGAI-EXAONE, LiquidAI, llm-jp, opendatalab, OpenHands, openvla, Salesforce, state-spaces, swiss-ai, TinyLlama, typhoon-ai

**Existing orgs with additional models**: allenai, arcee-ai, google, microsoft, Qwen

## Format

Both CSV files share the same format with three columns:

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
