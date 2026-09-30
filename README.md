# Vonk Forge models and recipes

## The recipe book at a glance

A **recipe** is one tested way to run a model on NVIDIA DGX Spark: the model files, the engine and its pinned software, the launch settings and the number of Sparks. Pick a model family, then an engine and a Spark count.

- **Versions:** a recipe runs the upstream project's version when it publishes one (otherwise the recipe's own), with the release date. Only the newest revision is kept, refreshed in place; exact pins (commits, image digests, file hashes) sit underneath.
- **Options:** settings such as context length are declared per recipe and say whether changing them needs a restart or a rebuild.
- **Licences are informational:** the licence and any regional limits are shown so you can decide; they never block a recipe.
- **Full catalog:** browse everything at [vonkforge.ai/recipes](https://vonkforge.ai/recipes).

<!-- overview:start -->
We cover Agents-A1, DeepSeek V4 Flash, DiffusionGemma, FastContext, FLUX.2, Gemma, GLM, Google Gemma, Hunyuan3D, HunyuanOCR, HunyuanVideo, HunyuanVideo Foley, Hy3 kodelow, Inkling, Laguna, Leanstral, LFM2.5, Ling, LTX, Meta Llama, MiMo, MiniMax H3, MiniMax M2, MiniMax M3, MOSS-VL, MOVA, Muse Glimmer, Nemotron, OpenAI gpt-oss, Ornith, Pixal3D, Qwen (text and vision), Qwen Image, SkinTokens, Step 3.7 Flash, Step1X-3D, TRELLIS, TripoSG, UI-Mate, UkisAI Swift Qwen3.8 checkpoints, VibeThinker, Wan. This section is generated from `recipes/`, `models/` and [`creators.json`](creators.json) by `tools/build-readme-overview`; do not edit it by hand.

### Engines and models

Which engines each model family runs on.

| Model family | vLLM | SGLang | TensorRT-LLM | llama.cpp | TensorFold | ds4 | diffusers | ComfyUI | pytorch-pipeline |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agents-A1 | ✓ |  |  |  |  |  |  |  |  |
| DeepSeek V4 Flash | ✓ | ✓ |  |  |  | ✓ |  |  |  |
| DiffusionGemma | ✓ |  |  |  |  |  |  |  |  |
| FastContext | ✓ |  |  |  |  |  |  |  |  |
| FLUX.2 |  |  |  |  |  |  |  | ✓ |  |
| Gemma | ✓ |  |  | ✓ |  |  |  |  |  |
| GLM | ✓ |  |  |  |  |  |  |  |  |
| Google Gemma | ✓ |  |  |  |  |  |  |  |  |
| Hunyuan3D |  |  |  |  |  |  |  |  | ✓ |
| HunyuanOCR |  |  |  |  |  |  |  |  | ✓ |
| HunyuanVideo |  |  |  |  |  |  | ✓ |  |  |
| HunyuanVideo Foley |  |  |  |  |  |  |  |  | ✓ |
| Hy3 kodelow | ✓ |  |  |  |  |  |  |  |  |
| Inkling | ✓ | ✓ |  |  |  |  |  |  |  |
| Laguna | ✓ |  |  |  |  |  |  |  |  |
| Leanstral | ✓ |  |  |  |  |  |  |  |  |
| LFM2.5 | ✓ |  |  |  |  |  |  |  |  |
| Ling | ✓ | ✓ |  |  |  |  |  |  |  |
| LTX |  |  |  |  |  |  | ✓ |  | ✓ |
| Meta Llama | ✓ | ✓ | ✓ |  |  |  |  |  |  |
| MiMo | ✓ | ✓ |  |  |  |  |  |  |  |
| MiniMax H3 |  |  |  |  |  |  | ✓ |  |  |
| MiniMax M2 | ✓ |  |  |  |  |  |  |  |  |
| MiniMax M3 | ✓ |  |  |  |  |  |  |  |  |
| MOSS-VL |  |  |  |  |  |  |  |  | ✓ |
| MOVA |  |  |  |  |  |  |  |  | ✓ |
| Muse Glimmer | ✓ |  |  |  |  |  |  |  |  |
| Nemotron | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |
| OpenAI gpt-oss |  |  | ✓ |  |  |  |  |  |  |
| Ornith | ✓ |  |  |  |  |  |  |  |  |
| Pixal3D |  |  |  |  |  |  |  |  | ✓ |
| Qwen (text and vision) | ✓ | ✓ |  | ✓ | ✓ |  |  |  |  |
| Qwen Image |  |  |  |  |  |  | ✓ | ✓ |  |
| SkinTokens |  |  |  |  |  |  |  |  | ✓ |
| Step 3.7 Flash | ✓ |  |  |  |  |  |  |  |  |
| Step1X-3D |  |  |  |  |  |  |  |  | ✓ |
| TRELLIS |  |  |  |  |  |  |  |  | ✓ |
| TripoSG |  |  |  |  |  |  |  |  | ✓ |
| UI-Mate | ✓ |  |  |  |  |  |  |  |  |
| UkisAI Swift Qwen3.8 checkpoints |  |  |  |  | ✓ |  |  |  |  |
| VibeThinker | ✓ |  |  |  |  |  |  |  |  |
| Wan |  |  |  |  |  |  |  | ✓ | ✓ |

### Spark variants

Which Spark counts each model family runs on.

| Model family | 1 Spark | 2 Sparks | 3 Sparks | 4 Sparks | 8 Sparks |
| --- | :---: | :---: | :---: | :---: | :---: |
| Agents-A1 | ✓ |  |  |  |  |
| DeepSeek V4 Flash | ✓ | ✓ | ✓ | ✓ |  |
| DiffusionGemma | ✓ |  |  |  |  |
| FastContext | ✓ |  |  |  |  |
| FLUX.2 | ✓ |  |  |  |  |
| Gemma | ✓ |  |  |  |  |
| GLM | ✓ | ✓ | ✓ | ✓ |  |
| Google Gemma | ✓ |  |  |  |  |
| Hunyuan3D | ✓ |  |  |  |  |
| HunyuanOCR | ✓ |  |  |  |  |
| HunyuanVideo | ✓ |  |  |  |  |
| HunyuanVideo Foley | ✓ |  |  |  |  |
| Hy3 kodelow |  | ✓ |  |  |  |
| Inkling |  | ✓ |  |  | ✓ |
| Laguna | ✓ |  |  |  |  |
| Leanstral |  | ✓ |  |  |  |
| LFM2.5 | ✓ |  |  |  |  |
| Ling | ✓ |  |  |  |  |
| LTX | ✓ |  |  |  |  |
| Meta Llama | ✓ |  |  |  |  |
| MiMo |  | ✓ | ✓ | ✓ |  |
| MiniMax H3 | ✓ |  |  |  |  |
| MiniMax M2 |  | ✓ |  |  |  |
| MiniMax M3 |  | ✓ | ✓ |  |  |
| MOSS-VL | ✓ |  |  |  |  |
| MOVA | ✓ |  |  |  |  |
| Muse Glimmer | ✓ |  |  |  |  |
| Nemotron | ✓ |  |  |  |  |
| OpenAI gpt-oss | ✓ |  |  |  |  |
| Ornith | ✓ |  |  |  |  |
| Pixal3D | ✓ |  |  |  |  |
| Qwen (text and vision) | ✓ | ✓ |  | ✓ |  |
| Qwen Image | ✓ |  |  |  |  |
| SkinTokens | ✓ |  |  |  |  |
| Step 3.7 Flash |  | ✓ |  |  |  |
| Step1X-3D | ✓ |  |  |  |  |
| TRELLIS | ✓ |  |  |  |  |
| TripoSG | ✓ |  |  |  |  |
| UI-Mate | ✓ |  |  |  |  |
| UkisAI Swift Qwen3.8 checkpoints | ✓ |  |  |  |  |
| VibeThinker | ✓ |  |  |  |  |
| Wan | ✓ |  |  |  |  |

### Tracked creators

Creators whose DGX Spark work we package; filter by creator in vonkctl or the web Library.

| Creator | Focus |
| --- | --- |
| [0xSero](https://github.com/0xSero) | SparkInfer builds and local-ai-recipe-kit; DeepSeek V4 Flash on one Spark. |
| [antirez](https://github.com/antirez/ds4) | ds4: the DeepSeek 4 Flash inference engine (Metal, CUDA, ROCm). |
| [drowzeys (keyz)](https://github.com/drowzeys) | Large multi-Spark vLLM builds with prebuilt images: GLM, MiMo, abliterated variants. |
| [eugr](https://github.com/eugr) | spark-vllm-docker: vLLM container builds for dual DGX Spark; llama-benchy benchmarking. |
| [MiaAI-Lab](https://github.com/MiaAI-Lab) | Spark cookbooks for SGLang, vLLM and TensorFold; DSpark and EXL3 builds; Qwen, GLM, DeepSeek, Ling. |
| [NVIDIA playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) | Official DGX Spark playbooks and NVIDIA model releases: Nemotron, NVFP4 checkpoints. |
| [r0b0tlab](https://github.com/r0b0tlab) | vLLM on GB10/SM121: EXL3 kernels, DFlash2 speculative decoding, GLM and Nemotron. |
| [sfxnz](https://github.com/sfxnz) | Dual-Spark vLLM TP=2 cookbooks: Qwen3.8, GLM 5.3, DeepSeek V4.1 EXL3. |
| [tonyd2wild](https://github.com/tonyd2wild) | Multi-Spark (2 and 4) vLLM recipes for large MoE models: GLM, MiMo. |
<!-- overview:end -->

## Contracts


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
| `requires_token` | Whether downloading the files needs a provider account token (a gated repository). The token itself stays in Controller secrets. |
| `dependencies` | Exact references to companion Models. |
| `format` | Numerical precision and quantization. |
| `license` | Usage terms, attribution and any territorial notice. |
| `files` | Each file’s ID, path, content hash, byte size and purpose, such as weights or tokenizer. |
| `capabilities` | The names of the features the model supports. |

Family, model, version and variant names are **data**, not Python classes. Adding a new family or version does not require changing the contract. File hashes and sizes live here once; recipes reference them.

[See a complete Model example →](contracts/src/vonk_forge_contracts/examples/model-definition.json)

## Recipe contract

[`RecipeDefinition`](contracts/src/vonk_forge_contracts/recipe.py) describes one way to run the selected model files.

| Field | Contents |
| --- | --- |
| `identity`, `metadata` | Publisher, unique name, title, description and tags. |
| `models` | Exact Model references, selected file IDs, and where each Spark role reads those files. |
| `execution` | How to build the container image: a pinned ARM64 base image, the build context, Dockerfile, patches and the hosts the build may reach. |
| `runtime` | Engine, launch command, arguments, environment and stop timeout. |
| `settings` | Generation, embedding or job settings, including whether changing a value needs a restart or rebuild. |
| `topology` | Number of Sparks, their roles, parallelism and start order, and each role's memory and disk needs. |
| `interfaces` | How an application uses the model: an API or a file-based job. |
| `validation` | Representative requests or job inputs and their expected results. |
| `release` | The version the recipe runs and its date: the upstream project's version when it publishes one, otherwise the recipe's own. |
| `provenance` | Where the recipe came from and who should be credited. |

Both documents declare their `kind` (`model` or `recipe`). Pydantic checks their structure; the shared resolver checks that a recipe references the right Models (by the `document_sha256` of their JSON) and files. Running the declared tests checks actual model behavior.

The contracts have one semantic version, `CONTRACT_VERSION`, which is also the version of the published library release (`v2.0.0`). Recipe and Model changes update that release in place; an additive contract change publishes a new minor release and a breaking one a new major release.

Each recipe is one JSON file. Its downloadable package adds the exact Model snapshots, build sources and test fixtures it needs; it contains no model weights or container images.

Examples: [build from source](contracts/src/vonk_forge_contracts/examples/recipe-source-build.json) · [two Sparks](contracts/src/vonk_forge_contracts/examples/recipe-dual.json) · [file-based job](contracts/src/vonk_forge_contracts/examples/recipe-job.json). These use synthetic data to show the structure.

## What the Controller handles

The repository contains definitions and build sources, not model weights or container images. The Controller caches model files and images separately on local storage, distributes them to the selected Sparks, and manages starting, stopping and progress. Runtime defaults such as writable engine caches belong to the platform, so every recipe does not have to repeat them.

## Where to look

- [Create and update recipes](docs/recipe-authoring.md): the standard workflow for agents and maintainers, including upstream refreshes and version notes.
- [Pydantic definitions](contracts/src/vonk_forge_contracts): the shared source of truth for Model and Recipe structure.
- [Generated JSON Schemas](contracts/src/vonk_forge_contracts/schema): editor and non-Python tooling support.
- [Contract guide](contracts/README.md): validation, package reuse and Controller integration details.
- [Public catalog](https://vonkforge.ai/recipes): browse models and recipes.

This repository defines the first-release catalog contract. Consumers use the same shared definitions; there is no parallel legacy catalog format.
