FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/FastContext-1.0-4B-RL-NVFP4" \
      org.opencontainers.image.revision="136de6a5cb5cd485ab57a0b8de373be409d9dc37" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# Parent: the vllm/vllm-openai digest already used by this catalog's vLLM recipes (vLLM 0.28.x). Upstream names no
# image, only "vLLM >= 0.23" with --quantization modelopt, so the engine pin is ours. This layer adds a cache
# launcher and the non-root user.
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
