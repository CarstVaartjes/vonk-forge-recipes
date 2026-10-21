# Vonk Forge public contracts

This package ships the Model, Recipe and qualification Pydantic models,
JSON Schemas and synthetic examples. See the [package API guide](../../README.md)
and the repository's [authoring guide](../../../docs/recipe-authoring.md).

Schemas are generated with `tools/generate-contract-schemas`; examples with
`tools/generate-contract-examples`. The repository gates check both against
the installed types and exercise an externally installed wheel.
