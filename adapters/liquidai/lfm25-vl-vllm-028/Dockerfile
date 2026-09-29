# syntax=docker/dockerfile:1.7
FROM docker.io/vllm/vllm-openai@sha256:e06f024cf1dd864a2f4b6e033b496807c55952404c14cf8c1b08e4ca9888597e

LABEL org.opencontainers.image.source="https://huggingface.co/LiquidAI/LFM2.5-VL-3B" \
      org.opencontainers.image.revision="a3af5799199acdd2a4f56ac4342816abb46c12a9" \
      org.opencontainers.image.licenses="LFM-1.0" \
      org.opencontainers.image.description="Offline vLLM 0.30 CUDA 13 ARM64 OpenAI adapter for LFM2.5-VL-3B" \
      io.vonk.vllm.build-commit="ced6857afa0ea7b2e3f0846a62e1394e90f15607"

RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /outputs /state

COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm

ENV HOME=/state \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    XDG_CACHE_HOME=/outputs/cache \
    XDG_CONFIG_HOME=/outputs/config \
    HF_HOME=/outputs/cache/huggingface \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    UV_CACHE_DIR=/outputs/cache/uv

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
