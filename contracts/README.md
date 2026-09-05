# Model and Recipe contracts

Vonk Forge has two public authoring documents: **Model** and **Recipe**. Their authoritative Pydantic definitions live in this package. Real catalog records live in `vonk-forge-recipes`; the Controller and public website consume them.

The schema describes the structure. Adding a family, model, version or quantization means adding data, not adding a Python subclass or editing an enum of model names. A model can declare several modalities.

## Model: what the model is

[`ModelDefinition`](src/vonk_forge_contracts/model.py) describes one exact model version and variant:

- A unique record identity, with family and logical model information for grouping, plus version and variant labels.
- Modalities, format, precision and quantization; parameter counts and applicable limits when known.
- Source repository and immutable revision, license and provenance.
- Access requirements, without credentials; official, derived or quantized lineage; exact companion Model references and an optional superseded Model reference.
- A canonical file manifest: file ID, relative path, SHA-256, exact byte length and purpose such as weights or tokenizer.
- Capability facts with their evidence status. Unknown support remains unknown. Capability evidence can come from a different source or revision than the weights.

The file manifest is the only source for file hashes and byte lengths. Download/cache totals are computed from it, including content deduplication. Recipes do not repeat these facts.

License terms, including territorial restrictions, are information for the user. They do not require Controller location settings or block downloads and runs. A provider may still require the user's account to have access and a token stored in Controller secrets.

See the [complete synthetic Model example](src/vonk_forge_contracts/examples/model-definition.json).

## Recipe: how to run it

[`RecipeDefinition`](src/vonk_forge_contracts/recipe.py) selects exact Model documents and the files needed by each topology role. A selector names a file in that model manifest and its read-only mount destination.

Execution has two mutually exclusive forms:

| Form | Author supplies | Platform supplies |
|---|---|---|
| Image | Final OCI image identity pinned by digest, target platform | Fetching, verified local caching, distribution and import |
| Source build | Pinned base image and recipe-owned source/context, Dockerfile and patches | Build execution and a receipt binding the final image to those exact inputs |

Model weights remain separate from container images. Image archive sizes and transfer receipts are produced by the platform, not copied into recipe authoring fields. A direct-image recipe needs no fake Dockerfile or build job.

The recipe also declares its runtime engine and launch intent, applicable settings, topology and resource envelope, serving interface, and representative tests. Generation, embedding and job settings have different typed structures; an image/audio/3D job does not need an invented LLM context length.

`release` keeps the version, date and changelog inside the same Recipe JSON file. Its `history` is newest first; the first entry matches the current version and date. Each entry records concise changes, optional upstream links, and an `upgrade_effect` of `none`, `restart`, `reprepare` or `rebuild`. Full document identity includes these notes; model-file and image identities let the Controller reuse cached bytes when only the notes change.

Engine invariants—such as vLLM writable cache paths—belong to the platform's engine implementation. Harness catalog entities, runtime-distribution documents and patch-bundle catalog entities are not extra documents the author maintains.

Examples: [direct image](src/vonk_forge_contracts/examples/recipe-image.json), [source build](src/vonk_forge_contracts/examples/recipe-source-build.json), [two Sparks](src/vonk_forge_contracts/examples/recipe-dual.json), [container job](src/vonk_forge_contracts/examples/recipe-job.json).

## Validation and serving tests

Validation has distinct responsibilities:

1. Pydantic validates strict types, required fields, mutually exclusive branches and relationships within the document.
2. The shared resolver checks exact Model references and selected file IDs. Package validation checks that required source and fixture paths belong to the self-contained recipe package.
3. The Controller resolves engine compatibility, capacity and executable images against the actual platform. Real serving and hardware tests observe runtime behavior.

Generated JSON Schema supports editors and non-Python consumers. It is not a replacement for semantic resolution or running a model. Passing a structural example does not establish physical Spark acceptance.

OpenAI checks declare an HTTP request. Container jobs declare filesystem fixture and output-slot bindings; `/outputs` is a directory, not an HTTP endpoint. Tests must exercise representative inference. Health alone is insufficient. Unknown assertions must fail validation, and accepted assertions must be enforced by the executor. Restart and cache reuse belong to the maintainer qualification run.

The examples use synthetic sources and image identities to illustrate the contract; they are not runnable model recommendations.

## Reuse and PostgreSQL

This is an ordinary reusable Python package; publishing to PyPI is unnecessary. Consumers follow the latest `main`, explicitly refresh it during CI/builds, and include it in their built application. Record the resolved commit for traceability. A lockfile must not silently prevent the requested latest-main refresh.

Catalog tooling and the Controller use the same validators and canonical serialization. The web consumes generated schemas/types. PostgreSQL stores validated canonical documents and digests, with query projections derived from those objects and database uniqueness/reference constraints. Pydantic does not automatically alter SQL tables.

Private API, Spark execution-plan, operation-progress, build-receipt and telemetry contracts remain platform-owned. They do not add public catalog-authoring documents.

## Release evidence

The shared definitions describe the supported first-release contract. Catalog validation, consumer integration, publication, and physical runtime checks establish separate facts. Report the exact commits and checks used; the existence of these types alone does not prove a successful deployment or model run.
