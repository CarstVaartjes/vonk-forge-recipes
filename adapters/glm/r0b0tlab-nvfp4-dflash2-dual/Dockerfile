# r0b0tlab's published image of his glm53-flash-nvfp4-sm121 kit (22269731, MIT): vLLM 0.28.1rc1.dev580 (vllm/vllm-openai
# sha256:b0501f99..., arm64, CUDA 13.0, FlashInfer 0.6.18) with the kit's four anchored SM121/GLM patches (SM90 NoPE MLA on
# capability 12, indexer top-k SM-count gate, DFlash2 aux capture, drafter KV group) applied, tag overlay-v2. The kit
# publishes and documents it (README "Run the published image"); this recipe runs that image as published, pinned by
# digest (the tag is mutable; it was resolved once to the digest below), and does not rebuild it from the kit's patches.
# This layer only adds the Controller placement wrapper at the platform's launcher path, the licence text and the
# non-root user. Base: the linux/arm64 image ghcr.io/r0b0tlab/glm53-flash-nvfp4-sm121, by digest.
FROM ghcr.io/r0b0tlab/glm53-flash-nvfp4-sm121:overlay-v2@sha256:b9edc8e0c3642a553f9178dd3139cf22b7b4bd4553a37d53f8fd49ac80ee06b0

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/glm53-flash-nvfp4-sm121" \
      org.opencontainers.image.revision="222697317a4a165767ab8ef17b44cf969a5ece09" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN { test -x /opt/vllm/.venv/bin/vllm || test -x /usr/local/bin/vllm; } \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
