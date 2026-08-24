FROM ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-GLM5.2-Quantrio-INT4-INT8-Mixed-Abliterated-C1-30toks-4x-DGX-Sparks" \
      org.opencontainers.image.revision="296699b44e024de938cb8b3251d49d4b1306a507" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0 AND BSD-3-Clause"

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY install-nccl.py verify-runtime.py /opt/vonk/build/

RUN python3 /opt/vonk/build/install-nccl.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && install --directory --owner 10001 --group 10001 /state /outputs \
    && test -x /opt/vonk/bin/vllm \
    && test -r /opt/vonk/lib/libnccl.so.2

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
