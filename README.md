# Vonk Forge recipes

Public Model and Recipe documents for running pinned model workloads on NVIDIA
DGX Spark, with shared Pydantic contracts, reproducible recipe packages and
qualification fixtures.

- [Working in this repository](AGENTS.md): setup, checks, CI and hardware sweep.
- [Create and update recipes](docs/recipe-authoring.md): the complete authoring guide.

## Contributor quickstart

```bash
git clone https://github.com/CarstVaartjes/vonk-forge-recipes && cd vonk-forge-recipes
# place the build-context adapter sources (vendored files live in the adapter-sources release)
uv run --python 3.14 --no-project --with-editable contracts python tools/fetch-adapter-sources
# run the tests: the exact environment and static checks are in AGENTS.md ("Use a task-local writable uv cache")
```

- **Add or change a recipe:** follow [docs/recipe-authoring.md](docs/recipe-authoring.md).
- **Add a large vendor file:** `tools/adapter-sources add <path>... [--origin URL]`,
  then commit the updated `adapter-sources.json` (details in [AGENTS.md](AGENTS.md)).
- **Open a PR** against `main` from a branch; run the checks listed in
  [AGENTS.md](AGENTS.md) first. CI runs the full suites on the integration batch.
