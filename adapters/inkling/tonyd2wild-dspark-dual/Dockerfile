FROM docker.io/eugr/spark-vllm@sha256:adc80460cea286f1f9bf96d399530096b491c27d8a153558f961a90fb7b655f0

LABEL org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.revision="60dee5b47ab14201381d486c6e994b5cf0e648a1" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# tonyd2wild's Inkling DSpark recipe runs eugr's published image (digest above, built from
# eugr/spark-vllm-docker at the revision above) and applies mods with `docker cp` and
# `docker exec` at container start. Apply the same mods at build time instead, in the
# recipe's order. eugr's inkling-sm12-paged-kv and instanttensor-hybrid-draft-loader are
# byte-identical to eugr's commit; the five inkling-* mods and prepare-dspark-overlay.py
# are tonyd2wild's (from tonyd2wild/Inkling-Small-NVFP4-DSpark-BF16-KV-262K-2x-DGX-Spark at
# 7795dd3a; see NOTICE.upstream). drop-caches (a host page-cache loop) is left out.
# inkling-sm12-paged-kv checks for a CUDA device before installing; no GPU exists at
# build time, so use the mod's own test override that skips that preflight (the patch
# anchors are still checked). inkling-audio-deps pins its four audio wheels by version.
COPY mods /opt/vonk/build/mods
COPY prepare-dspark-overlay.py /opt/vonk/build/prepare-dspark-overlay.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN set -eu; \
    export INKLING_SM12_MOD_SKIP_RUNTIME_CHECK=1; \
    for mod in inkling-sm12-paged-kv inkling-agent-parser-fix inkling-audio-deps inkling-dspark-vllm inkling-warmup-cap inkling-prefill-admission-cap instanttensor-hybrid-draft-loader; do \
      bash "/opt/vonk/build/mods/$mod/run.sh"; \
    done

# The draft: upstream downloads RadixArk/Inkling-Small-DSpark-Preview at fcc210dc and runs
# prepare-dspark-overlay.py, which writes a vLLM-flavoured config.json (architecture
# Qwen3DSparkModel, n_predict and dspark_block_size 7) and symlinks the draft's other files
# next to it. Reproduce that here: fetch only the draft's config.json at the pinned revision
# (checked against the Model manifest digest) and point the source symlinks at the read-only
# draft mount, so the overlay resolves to the mounted weights at run time.
RUN set -eu; \
    mkdir -p /opt/vonk/draft-src; \
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/RadixArk/Inkling-Small-DSpark-Preview/resolve/fcc210dcf244cc2f550ef1f75507eb4146c17a57/config.json', '/opt/vonk/draft-src/config.json')"; \
    echo "f360cef0abbc5756fe3d173501ae2bc701ad56d5d124d970cf08a5b17af82ca0  /opt/vonk/draft-src/config.json" | sha256sum -c -; \
    for name in README.md dflash.py dspark.py model.safetensors; do ln -s "/models/draft/$name" "/opt/vonk/draft-src/$name"; done; \
    DSPARK_MODEL_DIR=/opt/vonk/draft-src DSPARK_OVERLAY_DIR=/opt/vonk/draft-overlay python3 /opt/vonk/build/prepare-dspark-overlay.py; \
    test -f /opt/vonk/draft-overlay/config.json; \
    rm -rf /opt/vonk/build

RUN test -x /opt/vonk/bin/vllm \
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
