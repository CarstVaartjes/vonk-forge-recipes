FROM ghcr.io/drowzeys/inkling-sglang-gb10@sha256:079775a8deb415d16b89a505df0795987add3395e6e63a42a885a712ccb1ddc3

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-SGlang-SM121-optimized-on-Two-DGX-Sparks" \
      org.opencontainers.image.revision="539a7747d030233e9ed845dab682a948ff0cbfb2" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's published :kvquant image (lmsysorg/sglang dev-inkling-dspark
# b7252cc6 + NCCL 2.30 + the NVFP4 KV-cache patches; arm64, CUDA 13). scripts/ and
# patches/ are upstream's launchers and patch diff, carried byte-identically for
# provenance; the recipe reproduces scripts/nvfp4-kv-boot.sh. This layer adds the
# placement wrapper and the non-root user.
COPY scripts /opt/vonk/upstream-scripts
COPY patches /opt/vonk/upstream-patches
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN test -x /opt/vonk/bin/sglang-serve \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
