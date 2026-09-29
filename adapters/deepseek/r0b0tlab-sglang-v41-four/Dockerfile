FROM ghcr.io/r0b0tlab/dsv41-flash-tp4-sglang-sm121@sha256:537a7bc04a1de6cdaf2086556984a9ee26d15f479dbb2e7799f88bd52a321740

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/dsv41-flash-tp4-sglang-sm121" \
      org.opencontainers.image.revision="430c812cf5e69b69e740756ea72fecdc1944db29" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's production image (config ID sha256:c010623e..., SGLang da64c5cb, arm64,
# adapter overlay v4 baked in; docker/Dockerfile in this directory is upstream's recipe for it).
# This layer only adds the placement wrapper and the non-root user.
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN test -d /opt/dsv41/adapter \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

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
