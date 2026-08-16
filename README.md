# Vonk Forge recipe library

This repository is the public standard library of Vonk Forge model recipes.
It contains declarative, reviewable recipe material—not model weights,
container layers, credentials, or fleet state.

## What lives here

- `model-groups/`, `models/`, and `model-versions/`: exact model identity and
  artifact provenance;
- `recipes/`: one immutable execution binding per topology, including the
  primary model version and any exact auxiliary model versions such as a
  LoRA, encoder, tokenizer, VAE, or upscaler;
- `runtime-distributions/` and `patch-bundles/`: exact runtime/patch identity
  selected by recipes;
- `model-targets/`: the research ledger, including candidate and blocked
  upstreams that are not runnable defaults;
- `adapters/`: public recipe build contexts when a recipe needs model-specific
  build behavior.

Execution harness implementations, schema authority, control-plane state,
installation, and Spark acceptance remain in
[`vonk-forge`](https://github.com/CarstVaartjes/vonk-forge). This repository
references harnesses by exact identity; it does not reimplement them.

## Development and production

`main` is the development library. A production installation consumes an
approved immutable release tag and the exact commit recorded in its import
receipt. A mutable
branch or tag such as `latest` is never execution authority.

Every pull request is validated against the platform repository's pinned v1
contract. The validator checks JSON schema and semantics, exact dependency
resolution, source-context boundaries, deterministic recipe identity, and a
credential scan. GitHub Actions is the only publication path.

## Adding a recipe

1. Add or revise the model entities and exact artifact metadata.
2. Select an existing built-in harness and pin the runtime distribution and
   optional patch bundle.
3. Add one recipe for one topology. Replicas and distributed ranks are not
   interchangeable.
4. If the runtime needs companion weights, add each companion as its own
   model-version entity and reference it through the recipe's exact
   `dependencies` list. Do not hide a mutable download or a second model
   family inside an adapter script.
5. For image, audio, video, mesh, or other input-dependent jobs, declare the
   interface input contract and the matching read-only `inputs` security mount.
   Inputs are supplied per job; recipes never use host paths or runtime URLs.
6. Keep source contexts deterministic and free of secrets; build-time patches
   must be applied and verified before an image is published.
7. Run structural, container, and Spark acceptance before changing a target to
   `accepted`.

To install the accepted entries into a Vonk Forge control plane, use the
platform repository's `scripts/import-recipe-library` command. It validates
this checkout first, uploads only the referenced public build contexts, and
records the exact Git commit in the local import receipt. Its default is a
dry-run; candidate recipes require an explicit opt-in.

Every non-blocked target in the ledger has at least one candidate recipe,
including the language, image, audio, video, and 3D targets. The six blocked
targets intentionally have no recipe and remain research-only. Candidate
recipes are structurally validated and visible to operators, but are not
accepted defaults until their ARM64 container build and Spark canary pass.
The DeepSeek DS4 and two-Spark official DSpark/Mia recipes remain candidates
until fresh physical acceptance; historical prototype evidence is deliberately
not reused.

The target ledger is maintained from primary sources: the model author's
repository and model card, exact Hugging Face revisions, and the selected
runtime project's official documentation. Community reports and independent
benchmarks are useful discovery signals, but they do not make a recipe
accepted. For example, the current video/audio survey records the open-weight
LTX-2.3, MOVA, and HunyuanVideo-Foley checkpoints as candidate recipes while
keeping them out of the accepted default catalog until an ARM64 container and
Spark canary exist. Qwen Image Layered demonstrates the isolated `/inputs`
contract and emits a multi-artifact layer result rather than an OpenAI
response.
