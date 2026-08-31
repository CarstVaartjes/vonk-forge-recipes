# syntax=docker/dockerfile:1.7
FROM docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c

LABEL org.opencontainers.image.source="https://huggingface.co/LiquidAI/LFM2.5-VL-3B" \
      org.opencontainers.image.revision="a3af5799199acdd2a4f56ac4342816abb46c12a9" \
      org.opencontainers.image.licenses="LFM-1.0" \
      org.opencontainers.image.description="Offline CUDA 13 ARM64 OpenAI adapter for LFM2.5-VL-3B" \
      io.vonk.vllm.build-commit="6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"

RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /outputs /state

COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm

ENV HOME=/state \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    XDG_CACHE_HOME=/outputs/cache

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
