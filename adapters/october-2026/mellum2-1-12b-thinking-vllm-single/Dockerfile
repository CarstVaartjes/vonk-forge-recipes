FROM docker.io/vllm/vllm-openai@sha256:fd33bec1578f312a01547cc6b9691a5154c3e8bcfb76d2b35acb657bfc4eee8f
LABEL org.opencontainers.image.source="https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/tree/92ddae9fc7665e9f801d141d2e5a6b2caf2460c4" \
      org.opencontainers.image.revision="92ddae9fc7665e9f801d141d2e5a6b2caf2460c4"
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
