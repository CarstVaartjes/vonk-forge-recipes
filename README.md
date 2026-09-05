# Vonk Forge models and recipes

This repository defines **what a model is** and **how to run it** in [Vonk Forge](https://vonkforge.ai). Authors write two kinds of JSON document, validated by the shared [Pydantic contracts](contracts/src/vonk_forge_contracts).

- **Model:** one exact model version and variant, its capabilities and its files.
- **Recipe:** the model files, software, settings and hardware needed to run it.

Several recipes can use the same Model—for example, with different engines or with one Spark versus two. A recipe can also use several Models when it needs companion weights.

## Model contract

[`ModelDefinition`](contracts/src/vonk_forge_contracts/model.py) describes a specific set of model files.

| Field | Contents |
| --- | --- |
| `identity` | Publisher and unique name; family, model, version and variant for browsing and grouping. |
| `metadata` | Description and tags. |
| `modalities` | The kinds of data the model handles: text, images, audio, video, 3D or embeddings. |
| `source` | Where the files come from, with an exact source revision. |
| `format` | File format, numerical precision and quantization. |
| `parameters`, `limits` | Model size and applicable limits, such as context length. |
| `license` | Usage terms and any required acknowledgement. |
| `files` | Each file’s ID, path, content hash, byte size and purpose, such as weights or tokenizer. |
| `capabilities` | Supported features and the evidence behind them; unknown support stays unknown. |
| `provenance` | Sources and attribution for the record. |

Family, model, version and variant names are **data**, not Python classes. Adding a new family or version does not require changing the contract. File hashes and sizes live here once; recipes reference them.

[See a complete Model example →](contracts/src/vonk_forge_contracts/examples/model-definition.json)

## Recipe contract

[`RecipeDefinition`](contracts/src/vonk_forge_contracts/recipe.py) describes one way to run the selected model files.

| Field | Contents |
| --- | --- |
| `identity`, `metadata` | Publisher, unique name, title, description and tags. |
| `models` | Exact Model references, selected file IDs, and where each Spark role reads those files. |
| `execution` | Either a ready-made container image identified by its content hash, or the source and build instructions for creating one. |
| `runtime` | Engine, launch command, arguments, environment and start/stop steps. |
| `settings` | Generation, embedding or job settings, including whether changing a value needs a restart or rebuild. |
| `topology` | Number of Sparks, their roles, how they work together, and memory, disk and network requirements. |
| `interfaces` | How an application uses the model: an API or a file-based job. |
| `validation` | Representative requests or job inputs, expected results and benchmark declarations. |
| `provenance` | Where the recipe came from and who should be credited. |

Both documents declare `schema_version: 2` and their `kind` (`model` or `recipe`). Pydantic checks their structure; the shared resolver checks that a recipe references the right Models and files. Running the declared tests checks actual model behavior.

Examples: [ready-made image](contracts/src/vonk_forge_contracts/examples/recipe-image.json) · [build from source](contracts/src/vonk_forge_contracts/examples/recipe-source-build.json) · [two Sparks](contracts/src/vonk_forge_contracts/examples/recipe-dual.json) · [file-based job](contracts/src/vonk_forge_contracts/examples/recipe-job.json). These use synthetic data to show the structure.

## What the Controller handles

The repository contains definitions and build sources, not model weights or container images. The Controller caches model files and images separately on local storage, distributes them to the selected Sparks, and manages starting, stopping and progress. Runtime defaults such as writable engine caches belong to the platform, so every recipe does not have to repeat them.

## Where to look

- [Create and update recipes](docs/recipe-authoring.md): the standard workflow for agents and maintainers, including upstream refreshes and version notes.
- [Pydantic definitions](contracts/src/vonk_forge_contracts): the shared source of truth for Model and Recipe structure.
- [Generated JSON Schemas](contracts/src/vonk_forge_contracts/schema): editor and non-Python tooling support.
- [Contract guide](contracts/README.md): validation, package reuse and Controller integration details.
- [Public catalog](https://vonkforge.ai/recipes): browse models and recipes.

The new contracts are available for review. Converting the existing catalog and wiring all consumers to them is still in progress.
