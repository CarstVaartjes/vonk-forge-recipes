FROM docker.io/vllm/vllm-openai@sha256:3f55dd62f38da189041ebfad2eeccd784246c3f32b29ea5a8cdc4d19f5c02994

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/vibethinker-3b-nvfp4" \
      org.opencontainers.image.revision="fcf0c6e87b57b6d0f5cbf7377c10ec910c49e1fe" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# Parent: vllm/vllm-openai tag v0.22.0-aarch64-ubuntu2404 (upstream's FROM), pinned by registry digest. Upstream's
# Dockerfile.upstream only adds a Hugging Face download at startup, which the platform replaces with the mounted
# model. This layer adds a cache launcher and the non-root user.
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
    VLLM_NO_USAGE_STATS=1

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
