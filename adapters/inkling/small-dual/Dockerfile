FROM docker.io/lmsysorg/sglang@sha256:bbedab8cbf2d209b00f48f1e96ef4e9b638b98771477fa14e0e70d62679f383b

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="a74222ef6e690f851e2e4ff1c0be7dc1357be313" \
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
