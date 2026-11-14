# Vonk Forge contracts package

The installable `vonk_forge_contracts` package supplies the public
`ModelDefinition`, `RecipeDefinition` and qualification contracts, and the
published catalog index (`catalog-index.json`) with its vocabulary and its
consumer reader. Its Pydantic models and generated JSON Schemas support producer
validation and consumers without importing the Controller or a runtime.

Use [the authoring guide](../docs/recipe-authoring.md) for the complete Model and
Recipe workflow, and [AGENTS.md](../AGENTS.md) for repository checks and release
verification.

API entry points:

- [Model types](src/vonk_forge_contracts/model.py) and
  [Recipe types](src/vonk_forge_contracts/recipe.py).
- [Recipe API facts](src/vonk_forge_contracts/api.py) (an optional sibling of the
  Recipe, like its display) and [their checks](src/vonk_forge_contracts/api_checks.py).
- [Resolution and package validation](src/vonk_forge_contracts/resolver.py).
- [Published words](src/vonk_forge_contracts/vocabulary.py): each capability's label,
  description and category and each status's meaning and verified flag (the single owner;
  the index publishes them).
- [Published index](src/vonk_forge_contracts/catalog_index.py) (`CatalogIndex`; its schema
  ships in every release) and [its consumer reader](src/vonk_forge_contracts/catalog_reader.py):
  `read_catalog` returns the entries this release reads plus typed problems. An unknown
  word outside the execution document (a capability, modality, alignment, API endpoint or
  format, or qualification word) keeps its entry as an `UnrecognisedWord`. An unreadable
  optional block (display, API facts, qualification, prebuilt image, measured memory) is
  dropped and reported while its entry stays. An entry whose document, package, digests or
  source path are malformed is skipped and reported. A wrong envelope or contract major, or
  a `content_sha256` that does not match its document, refuses the index
  (`CatalogIndexRefused`). `read_catalog_index` is the strict whole-index form.
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
