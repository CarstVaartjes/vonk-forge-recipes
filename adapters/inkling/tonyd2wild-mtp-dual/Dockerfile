FROM docker.io/eugr/spark-vllm@sha256:adc80460cea286f1f9bf96d399530096b491c27d8a153558f961a90fb7b655f0

LABEL org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.revision="73f01ceea6369b8d4451462f094c20cb10559c11" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# tonyd2wild's Inkling recipe runs eugr's published image (digest above, built from
# eugr/spark-vllm-docker at the revision above) and applies mods with `docker cp` and
# `docker exec` at container start. Apply the same mods at build time instead, in the
# recipe's order: eugr's inkling-sm12-paged-kv and instanttensor-hybrid-draft-loader, and
# tonyd2wild's inkling-agent-parser-fix (from tonyd2wild/Inkling-Small-NVFP4-MTP-262K-2x-
# DGX-Spark at 7913395c; see NOTICE.upstream). The drop-caches mod (a host page-cache
# loop) is a memory-pressure helper, not image content, and is left out. The
# inkling-sm12-paged-kv mod checks for a CUDA device before installing; no GPU exists at
# build time, so use the mod's own test override that skips that preflight (the patch
# anchors are still checked).
COPY mods /opt/vonk/build/mods
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN set -eu; \
    export INKLING_SM12_MOD_SKIP_RUNTIME_CHECK=1; \
    for mod in inkling-sm12-paged-kv inkling-agent-parser-fix instanttensor-hybrid-draft-loader; do \
      bash "/opt/vonk/build/mods/$mod/run.sh"; \
    done \
    && rm -rf /opt/vonk/build \
    && test -x /opt/vonk/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
