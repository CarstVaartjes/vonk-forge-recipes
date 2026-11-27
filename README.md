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
5. Keep large vendor files out of Git: see
   [Build-context files kept out of Git](docs/recipe-authoring.md#build-context-files-kept-out-of-git).
6. Run the local checks below.
7. Open a PR against `main`. CI runs the whole suite and every gate on each pull request
   ([CI and publication](AGENTS.md#ci-and-publication)). Do not commit generated
   outputs ([section 6](docs/recipe-authoring.md#6-generate-publish-and-report)).

## Local checks

Producer tools use Python 3.14 and `uv`. Fetch the build-context sources once:

```bash
uv run --python 3.14 --no-project --with-editable contracts python tools/fetch-adapter-sources
```

Then follow [Setup and producer gates](AGENTS.md#setup-and-producer-gates) for
the disposable test environment and the exact lint, format, type and test
commands, which mirror `validate.yml`.

## How a recipe is qualified

Qualification is derived from hardware evidence on an exact identity, never
an authored label: see
[Hardware qualification status](docs/recipe-authoring.md#hardware-qualification-status)
and, for the sweep, [AGENTS.md](AGENTS.md#hardware-sweep-operation).

## Licence

This repository's code and data are MIT licensed ([LICENSE](LICENSE)). Adapters
under `adapters/` that vendor creator code keep their creators' own upstream
licences (see the LICENSE and NOTICE files beside them, or the upstream named in
`vendored-upstream.json`). [upstream-licences.json](upstream-licences.json)
records each upstream's licence and the path of its text (often under
[licenses/](licenses/)), outside every build context. [NOTICE](NOTICE) lists the
upstreams with no licence and the licence requests opened with their creators
(the files keep shipping with credit meanwhile) and where the Corresponding Source of
the AGPL-licensed recipe images lives. Models are licensed by their creators, as shown on each model's
licence line (informational, never a gate).
