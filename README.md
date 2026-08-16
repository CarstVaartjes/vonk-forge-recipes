# Vonk Forge recipe library

This repository is the public standard library of Vonk Forge model recipes.
It contains declarative, reviewable recipe material—not model weights,
container layers, credentials, or fleet state.

## What lives here

- `model-groups/`, `models/`, and `model-versions/`: exact model identity and
  artifact provenance;
- `recipes/`: one immutable execution binding per topology;
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
4. Keep source contexts deterministic and free of secrets; build-time patches
   must be applied and verified before an image is published.
5. Run structural, container, and Spark acceptance before changing a target to
   `accepted`.

The initial accepted entries are the single-Spark DeepSeek DS4 recipe and the
two-Spark official DSpark/Mia recipe. The rest of the target ledger is clearly
marked as candidate or blocked until it has exact artifacts and evidence.
