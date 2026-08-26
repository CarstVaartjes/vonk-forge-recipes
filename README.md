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
- `recipe-releases/`: additive release metadata and bounded changelog history
  for each executable recipe;
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

`catalog-index.json` is a generated, digest-bound view of every public recipe,
its exact catalog-entity closure, and the immutable Git blobs in each build
context. It lets the control plane load catalog metadata in one request instead
of spending one unauthenticated GitHub API request per recipe. When an operator
chooses a recipe, the control plane verifies and materializes only that recipe's
dependencies and source bundle. Run `tools/build-catalog-index` after changing
recipes, recipe releases, entities, or adapters; CI rejects a stale index.

Release metadata deliberately lives beside, rather than inside, the strict
executable recipe-v1 document. This keeps the execution-authority digest stable
while allowing catalog clients to compare an installed recipe digest with the
current semantic version and show every intervening change. Every
`recipe-releases/<recipe-slug>.json` file contains newest-first history. Each
history item binds a semantic version to the exact recipe content SHA-256,
declares whether upgrading needs a restart, reinstall, or image rebuild, and
contains bounded operator-facing change notes. Append a history item whenever
the executable recipe digest changes; metadata-only edits do not mint a new
recipe version.

Public recipe digests that predate the release sidecars are retained as `0.x`
history rather than being discarded as unknown revisions. Their entries link
to the exact repository commit that published the digest and use conservative
upgrade semantics. Where an original semantic release timestamp was never
recorded, the changelog says so explicitly instead of inventing one.

## Adding a recipe

1. Add or revise the model entities and exact artifact metadata.
2. Select an existing built-in harness and pin the runtime distribution and
   optional patch bundle.
3. Add one recipe for one topology. Replicas and distributed ranks are not
   interchangeable.
4. Add its `recipe-releases/<recipe-slug>.json` sidecar. Start at `1.0.0`, bind
   it to the canonical recipe digest emitted in `catalog-index.json`, and keep
   all later release entries newest-first so older installations can see the
   complete upgrade changelog.
5. If the runtime needs companion weights, add each companion as its own
   model-version entity and reference it through the recipe's exact
   `dependencies` list. Do not hide a mutable download or a second model
   family inside an adapter script.
6. For image, audio, video, mesh, or other input-dependent jobs, declare the
   interface input contract and the matching read-only `inputs` security mount.
   Inputs are supplied per job; recipes never use host paths or runtime URLs.
7. Keep source contexts deterministic and free of secrets; build-time patches
   must be applied and verified before an image is published.
8. Run structural, container, and Spark acceptance before changing a target to
   `accepted`.

To install the accepted entries into a Vonk Forge control plane, use the
platform repository's `scripts/import-recipe-library` command. It validates
this checkout first, uploads only the referenced public build contexts, and
records the exact Git commit in the local import receipt. Its default is a
dry-run; candidate recipes require an explicit opt-in.

Every non-blocked target in the ledger has at least one installable candidate
recipe, including the language, image, audio, video, and 3D targets. Every
published recipe document declares a complete executable contract, so an
operator may explicitly opt in and attempt installation before qualification.
The five blocked targets intentionally have no recipe and remain research-only.
Installability does not imply acceptance: candidates remain outside the
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

The 2026-08-22 runtime audit moved the public media catalog to Diffusers 0.40
and the PyTorch 2.13 CUDA 13 ARM64 stack with ABI-compatible TorchAudio 2.11,
and replaced remaining vLLM nightly
bindings with the stable vLLM 0.27.1 distribution. Qwen3.8 27B BF16 and FP8
recipes are included as exact-revision candidates; they still require a Spark
container build and physical canary before acceptance.

## Upstream drift audit

Immutable pins make a recipe reproducible; they do not by themselves say that
the selected upstream is still current. `tools/check-upstream-drift` performs a
read-only comparison between catalog source revisions and the channels declared
in `upstream-watch.json`. It discovers model-version, runtime-distribution, and
patch-bundle sources, plus the immutable upstream reference in recipe
provenance. Artifact inventories are not queried file by file.

GitHub sources may use the repository URL, an optional `.git` suffix,
`/tree/<commit>`, `/commit/<commit>`, `/blob/<commit>/<path>`, or the catalog's
historical `@<commit>` form. Hugging Face sources may use
`namespace/repository`, the canonical model URL, `/tree/<commit>`, or
`/resolve/<commit>/<path>`. All embedded revisions must agree with the
document's explicit revision.

The default policy follows the provider's default branch. Stable Diffusers,
PyTorch, and vLLM distributions override that default with `latest-release`.
Use a per-entity `ref` override for a deliberately tracked branch, or `manual`
when upstream movement cannot be classified automatically. A different head is
reported as `advanced`, which requests review but does not claim that the newer
source is compatible or accepted.

Run the audit locally with:

```console
GITHUB_TOKEN=... tools/check-upstream-drift --fail-on-drift --output upstream-drift.json
```

The command exits zero when all automated watches are current, one when input
or provider verification fails, and two for reviewable drift when
`--fail-on-drift` is selected. It never rewrites recipes, entities, the catalog
index, or watch metadata. The separate `Audit upstream drift` workflow runs
weekly and on manual dispatch, uploads both table and JSON reports, and marks
only that scheduled audit as failed when review is needed. Upstream network
state is deliberately not part of the required pull-request validation path.
