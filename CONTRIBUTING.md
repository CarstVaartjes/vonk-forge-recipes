# Contributing

Thank you for helping Vonk Forge run more models on DGX Spark.

- **Report a problem with a recipe:** open a bug report issue with the recipe
  id, the catalog version and the number of Sparks.
- **Propose a recipe or a fix:** follow
  [Add or update a recipe](README.md#add-or-update-a-recipe) and open a pull
  request against `main`.
- **Report a security problem:** see [SECURITY.md](SECURITY.md); never in a
  public issue.

The rules live in two places only: [AGENTS.md](AGENTS.md) for the repository
process, CI and the sweep, and [docs/recipe-authoring.md](docs/recipe-authoring.md)
for everything about recipes. This page does not repeat them.

## Release notes

There is no hand-kept changelog. The publish workflow creates a GitHub Release
for each catalog version, and its notes name the source commit. The merged pull
requests up to that commit are the history of changes.

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
