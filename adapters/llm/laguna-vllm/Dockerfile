FROM docker.io/vllm/vllm-openai@sha256:82f1e4adb3afe50a4cab5bf00b069576c41765565f5eef51dcc98ece26bd7ee2

LABEL org.opencontainers.image.source="https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4" \
      org.opencontainers.image.revision="d32afde8b09af1539b49ff96ff5551c674485f8e" \
      org.opencontainers.image.licenses="OpenMDW-1.1" \
      io.vonk.vllm.build-commit="8efa13b700f1836657699cae2503dc2feab27fa0"

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
