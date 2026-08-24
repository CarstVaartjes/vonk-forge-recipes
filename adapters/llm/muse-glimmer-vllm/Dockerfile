FROM docker.io/vllm/vllm-openai@sha256:ae1de325b8ea670288c328460a95d3807838cd254e8e9961ea7f6a741ce2c563

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="99a10304dce8945119bd0b1a072297803c52a749" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.vllm.build-commit="99a10304dce8945119bd0b1a072297803c52a749"

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
