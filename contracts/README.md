# Model and Recipe contracts

Vonk Forge has two public authoring documents: **Model** and **Recipe**. Their authoritative Pydantic definitions live in this package. Real catalog records live in `vonk-forge-recipes`; the Controller and public website consume them.

The schema describes the structure. Adding a family, model, version or quantization means adding data, not adding a Python subclass or editing an enum of model names. A model can declare several modalities.

## Model: what the model is

[`ModelDefinition`](src/vonk_forge_contracts/model.py) describes one exact model version and variant:

- A unique record identity, with family and logical model information for grouping, plus version and variant labels.
- Modalities, precision and quantization.
- Source identity and license. Git-backed sources bind an immutable revision; a GitHub release source binds a release ID and the asset IDs for the files below.
- `requires_token`: whether downloading the files needs a provider account token (a gated repository). No credentials are stored in the document.
- Exact companion Model references.
- A canonical file manifest: file ID, relative path, SHA-256, exact byte length and purpose such as weights or tokenizer.
- `capabilities`: the names of the capabilities the model supports. A name is data, not an enum, so a newer catalog may add one.

The file manifest is the only source for file hashes and byte lengths. Download/cache totals are computed from it, including content deduplication. Recipes do not repeat these facts. A GitHub release locator names only the public release and asset; this source branch uses anonymous access and does not accept provider credentials. It does not replace the file's SHA-256 or byte length.

License terms, including territorial restrictions, are information for the user. They do not require Controller location settings or block downloads and runs. A provider may still require the user's account to have access and a token stored in Controller secrets.

See the [complete synthetic Model example](src/vonk_forge_contracts/examples/model-definition.json).

## Recipe: how to run it

[`RecipeDefinition`](src/vonk_forge_contracts/recipe.py) selects exact Model documents, by the `document_sha256` of their published JSON, and the files needed by each topology role. A selector names a file in that model manifest and its mount destination; model mounts are always read-only.

Every recipe builds its image (`execution.build`): the author supplies a digest-pinned base image (always the linux/arm64 image for DGX Spark), the recipe-owned build context, Dockerfile and patches, and `network.hosts`, the only hosts the build may reach (an empty list builds offline). The platform runs the build. Model weights remain separate from container images.

The recipe also declares its runtime engine, entrypoint, ordered arguments, environment and stop timeout, applicable settings, topology and resource envelope, serving interface, and representative tests. Generation, embedding and job settings have different typed structures; an image/audio/3D job does not need an invented LLM context length.

The topology declares `node_count`, its roles with their resources, parallelism and start order. Everything else follows from the node count: one node runs alone; more nodes share one connected fabric with a 200 Gbit/s floor, losing a rank withdraws the endpoint, recovery restarts the workers and then the entrypoint, and stopping starts with the endpoint owner. `RecipeTopology` exposes these as `mode`, `world_size`, `fabric_connectivity`, `fabric_minimum_bandwidth_mbps` and `stop_order`. Each role declares unified memory as `peak_bytes` plus `reserve_bytes`, and disk as image, artifact, working (staging plus caches) and safety-margin bytes. Job inputs are staged read-only under `/inputs`; jobs write outputs under `/outputs`.

Runtime arguments are ordered trusted process data. Their names and bounded JSON-safe values pass through unchanged, including options unfamiliar to the Controller; compatibility resolution may report whether the pinned engine can execute them, but it does not define an allowlist. Repeated names retain their input order. An empty string is passed as an empty argv token. `value: null` is reserved for a setting-bound argument and requires `setting`; it is not a literal null value. For literal booleans, `true` emits the presence flag and `false` omits it. Each serialized name or value is limited to 65,536 UTF-8 bytes, a rendered argv is limited to 1 MiB, and a recipe has at most 128 runtime arguments. Structured values use compact, deterministic JSON. Platform-owned environment, mounts, writable paths, and other security constraints remain authoritative and are checked separately.

`release` is `{version, released_at}`: the upstream project's version and release date when the upstream publishes versions (for example `1.6`, released 2026-09-17), otherwise the recipe's own semantic version. It carries no history; Git holds it.

Engine invariants—such as vLLM writable cache paths—belong to the platform's engine implementation. Harness catalog entities, runtime-distribution documents and patch-bundle catalog entities are not extra documents the author maintains.

Examples: [source build](src/vonk_forge_contracts/examples/recipe-source-build.json), [two Sparks](src/vonk_forge_contracts/examples/recipe-dual.json), [container job](src/vonk_forge_contracts/examples/recipe-job.json).

## Validation and serving tests

Validation has distinct responsibilities:

1. Pydantic validates strict types, required fields and relationships within the document.
2. The shared resolver checks exact Model references and selected file IDs. Package validation checks that required source and fixture paths belong to the self-contained recipe package.
3. The Controller resolves engine compatibility, capacity and executable images against the actual platform. Real serving and hardware tests observe runtime behavior.

Generated JSON Schema supports editors and non-Python consumers. It is not a replacement for semantic resolution or running a model. Passing a structural example does not establish physical Spark acceptance.

Optional fields with a `None` default accept omission or explicit `null`;
omit unused optional fields when sending or authoring a document. Required
fields must always be present, including `null` when their type allows it.
Preserve meaningful false, zero, empty values and engine-owned JSON values.

## Versions, identity and reading

`CONTRACT_VERSION` (`2.1.0`) is the semantic version of these contracts and of
the recipe library release that publishes the catalog (`v2.1.0`). Documents
carry no schema version. Recipe and Model changes never change the library
version; publication updates the release in place and records `updated_at`.
An additive contract change (a new optional field) is a minor version and a
new release (for example `v2.2.0`); a breaking change is a major version (`v3.0.0`).

A document's identity is `document_sha256`: the SHA-256 of its canonical JSON
(sorted keys, compact separators, UTF-8) exactly as published, not of a
re-serialized parse. Hash a document once when it is published or imported
and keep the digest with it. Recipe and Model references hold the referenced
Model's `document_sha256`, and the resolvers take Models keyed by it.

Consumers read published documents with `read_model` and `read_recipe`, which
ignore fields a newer minor contract added. Authoring and the catalog build
validate strictly, rejecting unknown fields.

OpenAI checks declare an HTTP request. Container jobs declare filesystem fixture and output-slot bindings; the fixture is staged as the input when the interface declares one, and `/outputs` is a directory, not an HTTP endpoint. Tests must exercise representative inference. Health alone is insufficient. Unknown assertions must fail validation, and accepted assertions must be enforced by the executor. Restart and cache reuse belong to the maintainer qualification run.

The examples use synthetic sources and image identities to illustrate the contract; they are not runnable model recommendations.

## Reuse

This is an ordinary reusable Python package; publishing to PyPI is unnecessary. The platform pins a contracts commit and follows the newest library release within that contract's major version. Catalog tooling and the Controller use the same validators; the Controller stores each published document as received, with its digest.

## Release evidence

The shared definitions describe the supported first-release contract. Catalog validation, consumer integration, publication, and physical runtime checks establish separate facts. Report the exact commits and checks used; the existence of these types alone does not prove a successful deployment or model run.
