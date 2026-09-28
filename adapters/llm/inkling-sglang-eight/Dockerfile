FROM docker.io/lmsysorg/sglang@sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="94602c9c2b7cbdb8efd5c52802dac6a1c180089e" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.inkling.recipe="thinkingmachines/Inkling-NVFP4@42a75a99a40eb2ba1e0717db6357a0bf15205044"

COPY --chmod=0755 sglang-wrapper.py /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py

RUN python3 /opt/vonk/build/verify-runtime.py \
    && install --directory --owner 10001 --group 10001 /state /state/torch /outputs \
    && rm -rf /opt/vonk/build

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
    TORCHINDUCTOR_CACHE_DIR=/state/torch \
    HOME=/state

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
