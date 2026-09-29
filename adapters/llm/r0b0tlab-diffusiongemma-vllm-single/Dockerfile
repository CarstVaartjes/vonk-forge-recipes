FROM ghcr.io/r0b0tlab/vllm-diffusiongemma-26b-nvfp4-sm121@sha256:2b79eabe322cef7c80c61910f2c9a349f99050a58b9ca8459eb63a66e7557fb7

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/diffusiongemma-26b-nvfp4-sm121-vllm" \
      org.opencontainers.image.revision="c255963962e71c0cb9d7a4baed4ab5c410459988" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's patched image tag spark-dgemma-patched (Spark day-0 vLLM wheel plus the DiffusionGemma
# patch set in third_party/, image ID sha256:6ae2b249...; docker/Dockerfile.spark-dgemma-patched in this
# directory is upstream's recipe for it), pinned by registry digest. This layer adds a launcher that
# creates the platform-owned cache directories, and the non-root user.
USER root
RUN install --directory --mode=0755 /opt/vonk/bin \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache \
    && printf '%s\n' '#!/bin/sh' 'set -eu' 'mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm \
    && test -x "$(command -v vllm)"

ENV HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    XDG_CONFIG_HOME=/outputs/cache/config \
    HF_HOME=/outputs/cache/huggingface \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache/flashinfer \
    FLASHINFER_CACHE_DIR=/outputs/cache/flashinfer \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_USE_V2_MODEL_RUNNER=1

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
