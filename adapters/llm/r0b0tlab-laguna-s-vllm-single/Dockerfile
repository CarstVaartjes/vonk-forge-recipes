FROM ghcr.io/r0b0tlab/vllm-laguna-s-2.1-nvfp4-sm121@sha256:4f8ba8a454fefc7b81f8e6ceafe46ed92ddf1f409a160556e5b1ea4daaaee80a

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/laguna-s-2.1-nvfp4-sm121-vllm" \
      org.opencontainers.image.revision="4cf44cf36b88b564031ddb19c41d7cb9cfe0f52a" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's production image (tag v0.25.1-gb10-k7-0761412: vLLM 0.25.1, torch 2.11.0+cu130,
# FlashInfer 0.6.15.dev20260712, arm64), pinned by registry digest. docker/Dockerfile.production in this
# directory is upstream's recipe for it. This layer only adds a launcher that creates the platform-owned
# cache directories, and the non-root user.
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
    CUDA_CACHE_PATH=/outputs/cache/cuda

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
