FROM ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121@sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/xyz-aquila-mini-nvfp4-sm121-vllm" \
      org.opencontainers.image.revision="da18a86708bb2b1e2e455dd91685ce14a1a5ff17" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's vLLM v0.25.0 CUDA 13 arm64 SM121 image (tag v0.25.0-cu130-sm121-arm64-702f4814-r2), the same
# digest upstream's Dockerfile.upstream builds on. The ModelOpt mixed-precision patch below is upstream's, applied
# unchanged; this layer adds a launcher that creates the platform-owned cache directories, and the non-root user.
USER root
COPY patches/patch_modelopt_w4a16_native_w4a4.py /tmp/patch_modelopt_w4a16_native_w4a4.py
RUN install --directory --mode=0755 /opt/vonk/bin \
    && MODELOPT_PATH="$(python3 -c 'import importlib.util; s=importlib.util.find_spec("vllm.model_executor.layers.quantization.modelopt"); assert s and s.origin; print(s.origin)')" \
    && python3 /tmp/patch_modelopt_w4a16_native_w4a4.py --path "${MODELOPT_PATH}" \
    && python3 -m py_compile "${MODELOPT_PATH}" \
    && rm -f /tmp/patch_modelopt_w4a16_native_w4a4.py \
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
