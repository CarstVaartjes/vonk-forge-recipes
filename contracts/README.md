# Vonk Forge contracts package

The installable `vonk_forge_contracts` package supplies the public
`ModelDefinition`, `RecipeDefinition` and qualification contracts. Its Pydantic
models and generated JSON Schemas support producer validation and consumers
without importing the Controller or a runtime.

Use [the authoring guide](../docs/recipe-authoring.md) for the complete Model and
Recipe workflow, and [AGENTS.md](../AGENTS.md) for repository checks and release
verification.

API entry points:

- [Model types](src/vonk_forge_contracts/model.py) and
  [Recipe types](src/vonk_forge_contracts/recipe.py).
- [Resolution and package validation](src/vonk_forge_contracts/resolver.py).
- [Canonical document identity](src/vonk_forge_contracts/canonical.py).
- [Qualification authority](src/vonk_forge_contracts/qualification_authority.py),
  [native decisions](src/vonk_forge_contracts/systemone.py) and
  [local sweep observations](src/vonk_forge_contracts/sweep.py).
- [Synthetic examples](src/vonk_forge_contracts/examples) and
  [generated schemas](src/vonk_forge_contracts/schema).

Install from `contracts/` with pip or uv; no PyPI publication is required.
`read_model` and `read_recipe` are published-document consumer entry points;
`ModelDefinition.model_validate` and `RecipeDefinition.model_validate` are the
strict producer entry points. `document_sha256` hashes published JSON, and
`validate_recipe_models` resolves exact Models keyed by that digest.
