FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/muse-glimmer-30b-nvfp4-vllm" \
      org.opencontainers.image.revision="d22c7324f72fd6cdb95f54763d2bf2e2b7aac814" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# Upstream serves this checkpoint from a source build of vllm-project/vllm PR #51655
# (Muse support, head 1f7f0715, plain sm_120 cubins). Muse support has since shipped in
# the official vLLM 0.30.0 ARM64 release image, which the catalog's Muse BF16 recipe
# already uses; this recipe runs r0b0tlab's NVFP4 checkpoint and flags on that image.
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
