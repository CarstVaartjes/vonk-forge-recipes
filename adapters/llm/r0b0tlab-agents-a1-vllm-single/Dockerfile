FROM ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121@sha256:6e2dfa4c48ac73965890a864dc1510017f42981e743b3f2834b4b189ad7712a8

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/agents-a1-nvfp4-sm121-vllm" \
      org.opencontainers.image.revision="fda20c0526e387da1446c8880a69cbda227fc851" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: the public laguna-m1 image digest (tag cu130-sm121-arm64-dda4668b, vLLM dda4668b, arm64) that upstream's
# Dockerfile.upstream names as its mutable :latest base. The fallback FlashAttention/Marlin removal and the
# vllm_flash_attn import relaxation below reproduce upstream's derivative layer (docker/Dockerfile.upstream);
# this layer also adds a cache launcher and the non-root user.
USER root
COPY strip_fa_check.py /tmp/strip_fa_check.py
RUN install --directory --mode=0755 /opt/vonk/bin \
    && rm -f /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C*.so /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa3_C*.so \
    && find /usr/local/lib/python3.12/dist-packages -iname '*marlin*.so' -delete \
    && python3 /tmp/strip_fa_check.py \
    && rm -f /tmp/strip_fa_check.py \
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
