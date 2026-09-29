FROM docker.io/vllm/vllm-openai@sha256:505aeb664ac0081f91e3e3d20296637b6ad263b5090b0ac56153e565cd01d67e

# Upstream tracks the moving tag vllm/vllm-openai:nightly-aarch64; this is its
# arm64 digest on 2026-09-29 (vLLM build commit below).
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000" \
      org.opencontainers.image.revision="608a48de75ba059fc9415947ad636a123935da15" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.vllm.build-commit="36768d1bfd39094681cdbc8cb37d4b31c0729c89"

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /opt/vonk/bin \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache \
    && printf '%s\n' '#!/bin/sh' 'set -eu' 'mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR"' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

ENV HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    XDG_CONFIG_HOME=/outputs/cache/config \
    HF_HOME=/outputs/cache/huggingface \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    UV_CACHE_DIR=/outputs/cache/uv

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
