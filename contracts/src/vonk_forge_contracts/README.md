# Vonk Forge public contracts

`ModelDefinition` and `RecipeDefinition` are the only author-facing roots in
this package. They are strict Pydantic v2 models with pure semantic checks and
no Controller, runtime, or platform imports. The checked-in JSON Schemas under
`contracts/schema/` are generated from these roots:

```bash
tools/generate-contract-schemas
tools/generate-contract-schemas --check
```

The package is consumed in-repository during this greenfield transition. The
platform and catalog builds should refresh the package from the latest
`vonk-forge-recipes` `main` source before frozen dependency resolution, then
record the resolved Git commit in build metadata. A manually maintained source
revision pin is intentionally unnecessary; contract changes trigger dependent
schema, catalog, and Controller checks. A future wheel can use the same
package without changing the document authority.
