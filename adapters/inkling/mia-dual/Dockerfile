FROM ghcr.io/drowzeys/inkling-sglang-gb10@sha256:079775a8deb415d16b89a505df0795987add3395e6e63a42a885a712ccb1ddc3

# Mia's Inkling Small dual-Spark kit runs drowzeys' prebuilt KV-quant GB10 image
# (SGLang dev-inkling-dspark build b7252cc6 plus fp4_mx_block16 KV, DSpark
# conv-commit and draft-context patches), pulled by tag in start_sglang.sh; we
# pin that image by digest and only add the Vonk launcher and runtime user.
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Inkling-Small-NVFP4-Dual-DGX-Sparks" \
      org.opencontainers.image.revision="ce6bfbe90d6e680e7b8172ebd1922fa148158e26"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py

RUN python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && test -x /opt/vonk/bin/sglang-serve

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
