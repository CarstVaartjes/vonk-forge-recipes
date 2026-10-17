FROM docker.io/vllm/vllm-openai@sha256:fd33bec1578f312a01547cc6b9691a5154c3e8bcfb76d2b35acb657bfc4eee8f
LABEL org.opencontainers.image.source="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tree/59b8721646c38111a242da18218859fb2b454322" \
      org.opencontainers.image.revision="59b8721646c38111a242da18218859fb2b454322"
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
