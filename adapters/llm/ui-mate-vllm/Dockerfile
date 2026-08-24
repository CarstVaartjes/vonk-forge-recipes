FROM docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c

LABEL org.opencontainers.image.source="https://github.com/Tencent/UI-Mate" \
      org.opencontainers.image.revision="d2b2e0aede83eeacfb1bc86f66503acbc4a6738a" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.vllm.build-commit="6e448d0ea9bf3d88d898b65449ca6dc2aec170ac" \
      io.vonk.ui-mate.upstream-archive-sha256="98d13c2f66cc128d5b96b3c574243d02adb89b27f9a4dab4aae5ef0a7bbf9f28"

COPY agents /opt/vonk/ui-mate/agents
COPY LICENSE.upstream NOTICE /opt/vonk/ui-mate/
COPY ui-mate-step.py vllm-wrapper.sh /opt/vonk/bin/

RUN chmod 0755 /opt/vonk/bin/ui-mate-step /opt/vonk/bin/vllm-wrapper.sh \
    && ln -s /opt/vonk/bin/vllm-wrapper.sh /opt/vonk/bin/vllm \
    && PYTHONPATH=/opt/vonk/ui-mate python -c \
      'import openai, PIL, requests; from agents.ui_mate_agent import UIMateAgent, parse_response; assert UIMateAgent and parse_response'

ENV PYTHONPATH=/opt/vonk/ui-mate \
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_API_KEY=EMPTY

USER 10001:10001
ENTRYPOINT []
