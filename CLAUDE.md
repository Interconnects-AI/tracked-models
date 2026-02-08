# Tracked Models - Development Guide

## CSV Format

`models.csv` has three columns: `org,model,modelId`

```csv
org,model,modelId
meta-llama,Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-8B-Instruct
```

## Reading the Model List in Python

```python
import csv

# From local file
with open('models.csv') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    models = [(org, model, model_id) for org, model, model_id in reader]

# From GitHub (raw URL)
import urllib.request, io
url = 'https://raw.githubusercontent.com/Interconnects-AI/tracked-models/main/models.csv'
with urllib.request.urlopen(url) as resp:
    text = resp.read().decode()
reader = csv.reader(io.StringIO(text))
next(reader)
models = [(org, model, model_id) for org, model, model_id in reader]

# With pandas
import pandas as pd
df = pd.read_csv('models.csv')
model_ids = df['modelId'].tolist()
```

## Writing / Modifying the CSV

When adding or removing models, preserve the existing row order to keep git diffs clean. Use Unix line endings (`\n`).

```python
import csv

def read_models(path):
    """Read models.csv and return list of (org, model, modelId) tuples."""
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        return [(row[0], row[1], row[2]) for row in reader if len(row) >= 3]

def write_models(path, rows):
    """Write models.csv with Unix line endings."""
    with open(path, 'w', newline='\n') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['org', 'model', 'modelId'])
        for row in rows:
            writer.writerow(row)

# Add new models preserving order (insert after last row of existing org)
def add_models(original, additions):
    """Insert additions after the last row of each matching org section."""
    from collections import OrderedDict
    adds_by_org = OrderedDict()
    for o, m, mid in additions:
        adds_by_org.setdefault(o, []).append((o, m, mid))

    result = []
    last_org = None

    for o, m, mid in original:
        if last_org is not None and o != last_org and last_org in adds_by_org:
            result.extend(adds_by_org.pop(last_org))
        result.append((o, m, mid))
        last_org = o

    # Last org in file
    if last_org in adds_by_org:
        result.extend(adds_by_org.pop(last_org))

    # New orgs appended at end, sorted
    for org in sorted(adds_by_org.keys(), key=str.lower):
        result.extend(adds_by_org[org])

    return result
```

## When Adding Models

1. **Update the changelog** in `README.md` with the date, count, and org names
2. **Preserve row order** - insert new models after the last row of each org section, append new orgs at end
3. **Use Unix line endings** (`lineterminator='\n'`) to avoid ugly diffs

## Inclusion Criteria

- **Post-ChatGPT only**: Released after Nov 30, 2022
- **First-party weights**: Original model weights on HuggingFace (no third-party quants/reuploads)
- **>100K total downloads** for new additions (exceptions for notable recent releases)
- **LLMs and VLMs only**: `text-generation` and `image-text-to-text` pipeline tags
- **No guard/shield models**: Safety classifiers are excluded (Llama-Guard, ShieldGemma, Qwen3Guard, granite-guardian, etc.)
- **No T5Gemma variants**: Disproportionate download counts relative to real-world impact

## Checking Model Eligibility

```python
from huggingface_hub import HfApi
from datetime import datetime

api = HfApi()
CHATGPT_DATE = datetime(2022, 11, 30)

def check_model(model_id):
    """Check if a model meets inclusion criteria."""
    info = api.model_info(model_id)

    # Check pipeline tag
    valid_tasks = {'text-generation', 'image-text-to-text'}
    if info.pipeline_tag not in valid_tasks:
        print(f"Skip: pipeline_tag={info.pipeline_tag}")
        return False

    # Check creation date (post-ChatGPT)
    if info.created_at < CHATGPT_DATE:
        print(f"Skip: pre-ChatGPT ({info.created_at.date()})")
        return False

    # Check downloads
    if info.downloads < 100_000:
        print(f"Note: only {info.downloads:,} downloads (30d)")

    print(f"OK: {model_id} ({info.pipeline_tag}, {info.created_at.date()}, {info.downloads:,} downloads)")
    return True
```

## Cross-Referencing with HF Download Data

The [open-model-analysis](https://github.com/Interconnects-AI/open-model-analysis) repo has scripts for querying the `interconnects/hf-dumps` dataset to find high-download models not yet tracked. See `hf_utils.py` and `query_hf_data.py` there.
