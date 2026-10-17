# Organization logos

`orgs/` holds one logo per model-building organization, named by its Hugging Face
organization slug (e.g. `orgs/deepseek-ai.webp`, `orgs/allenai.webp`). Use the same
slug as the `org` column in `models.csv` or `hf_org` in
`metadata/organization_regions.csv`; the extension varies (`.webp`, `.svg`, `.png`).

## Sources

- Most files are the organization's Hugging Face avatar, copied from the snapshot
  in [projectvail/artifacts-hub](https://github.com/projectvail/artifacts-hub)
  (`public/logos/orgs/`, fetched via `https://huggingface.co/api/organizations/<slug>/avatar`).
- `EleutherAI.webp` and `open-athena.webp` were fetched from the same Hugging Face
  avatar API when this directory was created.
- `huggingface.svg` is the transparent Hugging Face mark from artifacts-hub
  (`public/logos/huggingface.svg`), which reads better on slides than the avatar.

## Aliases

Some organizations publish under several slugs. Use the canonical logo:

| Slug | Logo |
|------|------|
| `THUDM` | `zai-org` |
| `facebook`, `meta-models` | `meta-llama` |
| `mistral-community` | `mistralai` |
| `CohereForAI` | `CohereLabs` |
| `AIDC-AI` | `ATH-MaaS` |
| `vikhyatk` | `moondream` |
| `EvolutionaryScale` | `biohub` |
| `homebrewltd` | `Menlo` |

## Adding or refreshing a logo

Fetch the avatar URL, then download it into `orgs/<slug>.<ext>`:

```bash
curl -s https://huggingface.co/api/organizations/<slug>/avatar
```

Hugging Face's avatar CDN may serve WebP whatever the URL's extension says, so
check the real type (`file orgs/<slug>.*`) and name the file to match.

## Usage

Logos are trademarks of their respective organizations. They are collected here
for identifying organizations in research charts, dashboards, and talks. They are
not covered by this repository's Apache-2.0 license.
