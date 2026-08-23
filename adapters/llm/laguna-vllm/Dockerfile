FROM docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c

LABEL org.opencontainers.image.source="https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4" \
      org.opencontainers.image.revision="d32afde8b09af1539b49ff96ff5551c674485f8e" \
      org.opencontainers.image.licenses="OpenMDW-1.1" \
      io.vonk.vllm.build-commit="6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
