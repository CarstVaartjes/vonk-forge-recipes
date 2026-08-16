FROM docker.io/vllm/vllm-openai@sha256:82f1e4adb3afe50a4cab5bf00b069576c41765565f5eef51dcc98ece26bd7ee2

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="8efa13b700f1836657699cae2503dc2feab27fa0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.vllm.build-commit="8efa13b700f1836657699cae2503dc2feab27fa0"

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
