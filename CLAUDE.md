# Claude Code Guidelines

## When Adding Models

When adding new models to `models.csv`:

1. **Update the changelog** in `README.md` with:
   - The date (YYYY-MM-DD format)
   - A summary of models added (organization/collection name and count)
   - List the specific model names

2. **Maintain alphabetical order** - Models in `models.csv` are sorted by organization name, then by model name within each organization.

3. **CSV format** - Each row has three columns: `org,model,modelId` where modelId is `org/model`.
