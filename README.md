# Tracked Models

A list of HuggingFace models tracked by [Interconnects AI](https://interconnects.ai) for open model analytics.

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

## License

Apache 2.0
