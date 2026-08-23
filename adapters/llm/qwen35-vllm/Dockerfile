FROM docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c

LABEL org.opencontainers.image.source="https://huggingface.co/Qwen/Qwen3.5-9B" \
      org.opencontainers.image.revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.vllm.build-commit="6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
