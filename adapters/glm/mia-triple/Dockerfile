FROM ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks@sha256:f8f350d46b33858eda4f0c0a5c39a7f0b27005111d393098baaaacae18c10fdb

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks" \
      org.opencontainers.image.revision="5c85163ccb8d98d395880e71e2dbd03976a3f4ad" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN install --directory --owner 10001 --group 10001 /state /outputs \
    && test -x /opt/vonk/bin/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
