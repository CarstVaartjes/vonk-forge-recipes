FROM docker.io/eugr/spark-vllm-b12x@sha256:52f8e6338c5038ebcbc9c58e52e1d78eaccaa263e1dc1726f4cbecdced49f384

LABEL org.opencontainers.image.source="https://github.com/mattcurfman/glm-5.2-quanttrio-200k-4x-dgx-spark" \
      org.opencontainers.image.revision="b6e57cec7ed1ed491693751fa5f6ca991d532045" \
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
