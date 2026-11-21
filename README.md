# Vonk Forge recipes

Public Model and Recipe JSON documents for running pinned model workloads on
NVIDIA DGX Spark, with the shared Pydantic contracts (`contracts/`), recipe
packaging and qualification definitions (`qualification/`). Recipes live in
`recipes/`, the Models they run in `models/`, adapter sources in `adapters/`.

Rules live in two places only: [AGENTS.md](AGENTS.md) (repository process, CI,
sweep) and [docs/recipe-authoring.md](docs/recipe-authoring.md) (everything about
recipes). This page only points to them.

## Add or update a recipe

1. Establish the change and check the real upstreams
   ([sections 1-2](docs/recipe-authoring.md#1-establish-the-change)).
2. Author the Model and the Recipe JSON, following the creator's own values
   ([section 3](docs/recipe-authoring.md#3-author-the-model-and-recipe)).
3. Add the qualification definition `qualification/recipes/<slug>.json`
   ([section 5](docs/recipe-authoring.md#5-validate-and-exercise-the-result)).
4. Version the change and say what changed
   ([section 4](docs/recipe-authoring.md#4-version-and-explain-what-changed)).
5. Put large vendor files behind `tools/adapter-sources add <path>...` and commit
   `adapter-sources.json`; Git never holds files over 1 MiB.
6. Run the local checks below.
7. Open a PR against `main`. CI runs the full suites on the integration batch
   ([CI and publication](AGENTS.md#ci-and-publication)). Do not commit generated
   outputs ([section 6](docs/recipe-authoring.md#6-generate-publish-and-report)).

## Local checks

Producer tools use Python 3.14 and `uv`. Fetch the build-context sources once:

```bash
uv run --python 3.14 --no-project --with-editable contracts python tools/fetch-adapter-sources
```

Then, in a disposable environment outside the checkout (exact setup in
[AGENTS.md](AGENTS.md#setup-and-producer-gates), mirroring `validate.yml`):

```bash
python -m pytest -q <targeted tests>   # CI runs the whole suite
tools/check-python-lint
tools/check-python-format
scripts/check-python-types
git diff --check
```

`scripts/install-git-hooks` installs the pre-commit hook, which runs the
lint, format, type and added-file-size checks.

## How a recipe is qualified

Qualification is derived from hardware evidence on an exact recipe, image and
Model identity; it is never an authored label. Evidence records live in
`qualification/evidence/`. A change to the recipe, image or Model means a
re-check, not a failure. Statuses and what they mean:
[Hardware qualification status](docs/recipe-authoring.md#hardware-qualification-status).
How the sweep runs on hardware: [AGENTS.md](AGENTS.md#hardware-sweep-operation).
