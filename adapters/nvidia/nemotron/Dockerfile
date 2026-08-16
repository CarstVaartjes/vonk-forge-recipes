FROM docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c

LABEL org.opencontainers.image.source="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" \
      org.opencontainers.image.revision="6dbbd757ea75a8ece6e0702872e3ae53f9987728"

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
