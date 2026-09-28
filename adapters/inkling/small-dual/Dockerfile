FROM docker.io/lmsysorg/sglang@sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="94602c9c2b7cbdb8efd5c52802dac6a1c180089e" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.sglang.upstream-profile="docs/src/snippets/configs/thinkingmachines/inkling-small.jsx"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py

RUN python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && test -x /opt/vonk/bin/sglang-serve

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
