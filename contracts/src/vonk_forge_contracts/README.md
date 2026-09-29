# Vonk Forge public contracts

`ModelDefinition` and `RecipeDefinition` are the two author-facing roots in
this package; the qualification authority is the maintainer campaign's input.
All are strict Pydantic v2 models with pure semantic checks and no Controller,
runtime, or platform imports. The checked-in JSON Schemas under `schema/` are
generated from these models:

```bash
tools/generate-contract-schemas
tools/generate-contract-schemas --check
```

`CONTRACT_VERSION` is the semantic version of the Model and Recipe contracts
and of the recipe library release that publishes them. An additive change is a
minor release and a breaking change a major one. Model and Recipe documents
carry no schema version of their own. Consumers read published documents with
`read_model`/`read_recipe` and identify each by the `document_sha256` of its
published JSON.
