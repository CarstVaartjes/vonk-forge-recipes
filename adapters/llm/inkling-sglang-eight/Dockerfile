FROM docker.io/lmsysorg/sglang@sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19" \
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
