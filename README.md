# Tracked Models

The core HuggingFace models used to track metrics for [The ATOM Project](https://www.atomproject.ai/).

## Format

`models.csv` contains three columns:

| Column | Description |
|--------|-------------|
| `org` | HuggingFace organization/user |
| `model` | Model name |
| `modelId` | Full model identifier (`org/model`) |

## Usage

```bash
# Fetch the raw CSV
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/models.csv

# Get just the model IDs
curl -s https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/models.csv | tail -n +2 | cut -d',' -f3
```

## Changelog

### 2026-01-09
- Added arcee-ai Trinity collection (8 models): Trinity-Mini, Trinity-Mini-Base, Trinity-Mini-Base-Pre-Anneal, Trinity-Mini-GGUF, Trinity-Nano-Base, Trinity-Nano-Base-Pre-Anneal, Trinity-Nano-Preview, Trinity-Nano-Preview-GGUF
- Added arcee-ai AFM 4.5B series (5 models): AFM-4.5B, AFM-4.5B-Base, AFM-4.5B-GGUF, AFM-4.5B-ov, AFM-4.5B-Preview

## License

Apache 2.0
