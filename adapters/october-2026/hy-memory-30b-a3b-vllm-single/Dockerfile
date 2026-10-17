FROM docker.io/vllm/vllm-openai@sha256:fd33bec1578f312a01547cc6b9691a5154c3e8bcfb76d2b35acb657bfc4eee8f
LABEL org.opencontainers.image.source="https://huggingface.co/tencent/Hy-Memory-30B-A3B/tree/64a460a94d5498b7316fb02395c72d01c47a7347" \
      org.opencontainers.image.revision="64a460a94d5498b7316fb02395c72d01c47a7347"
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
