# inkling-dspark-vllm mod

This startup mod patches the pinned vLLM Inkling NVIDIA model so an external
EAGLE3/DSpark draft can consume auxiliary post-layer hidden states.

It is intentionally narrow and fail-closed:

- checks the expected upstream Python structure before editing;
- applies idempotently;
- adds `SupportsEagle3` and `EagleModelMixin`;
- exposes the tapped logical post-layer state while preserving Inkling's fused
  short-convolution path on untapped layers;
- syntax-checks the patched module before startup continues.

Verified against the vLLM image and upstream runtime revisions documented in
the repository README. Revalidate the anchors before using it with another
vLLM build.
