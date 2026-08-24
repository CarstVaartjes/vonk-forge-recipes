FROM docker.io/vllm/vllm-openai@sha256:871fc9b75a97dc6f58449bfb2aa30cab685c87a8bf6b4519a75c4760c1f8bcd8

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="88d34c6409e9fb3c7b8ca0c04756f061d2099eb1" \
      org.opencontainers.image.version="0.20.0"

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
