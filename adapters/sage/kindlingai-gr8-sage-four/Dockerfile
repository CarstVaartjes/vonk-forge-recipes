# KindlingAI's Sage-S2.1-Flash exlr8 gr8 recipe for four DGX Sparks at commit 1f106277 (MIT), on gr8's published image of
# the kit's gr8 submodule commit ca7e8745 (the image's revision label): gr8's from-scratch CUDA engine (g6m4_flash-int4,
# which runs TP widths 1 to 4), its Rust front end, the loader and the launcher on nvidia/cuda 13.0.1 (arm64). The kit
# publishes no image of its own, and its Dockerfile's ENGINE_SRC=image path takes the binaries from this image unchanged;
# this layer is that path's runtime stage: the kit's knobs, entrypoint, launcher shim and prompt-cache file at the kit's
# paths and its environment, plus the Controller placement wrapper at the platform's launcher path, the licence text and
# the non-root user. Nothing compiles. Base: the linux/arm64 image ghcr.io/kindlingai/gr8, by digest.
FROM ghcr.io/kindlingai/gr8:sha-ca7e874@sha256:d140aca32886d2055b7ff11d88c8860c6bb9c108c626f8f3a4f3cebdfe5fcfc6

LABEL org.opencontainers.image.source="https://github.com/kindlingai/sage-s2.1-flash-exlr8-4x-spark" \
      org.opencontainers.image.revision="1f1062775e075c1746c97cf64b38fa8f3046cd1c" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY LICENSE /opt/vonk/licenses/LICENSE.creator
COPY kv-store.yaml /etc/gr8/kv-store.yaml
COPY --chmod=0755 entrypoint.sh /opt/sage-4x/entrypoint.sh
COPY knobs.sh /opt/sage-4x/knobs.sh
COPY --chmod=0755 bin/gr8 /opt/sage-4x/bin/gr8
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/docker/entrypoint.sh && test -x /opt/gr8/bin/g6m4_flash-int4 \
    && test -x /opt/gr8/bin/g6load && test -x /opt/gr8/bin/g6front-rs && test -f /opt/gr8/bin/lib/libgr8knobs.so \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The kit's runtime environment (its Dockerfile's ENV lines, less MODEL: the wrapper serves the mounted release in
# place). The kit keeps its working files in /var/lib/gr8 volumes; the wrapper moves them under /outputs/cache/gr8, the
# platform's writable state.
ENV PATH=/opt/sage-4x/bin:/opt/gr8/bin:$PATH PYTHONPATH=/opt/gr8/frontend GR8_CONTAINER=1 GR8_MODEL=glm53f \
    PY=/usr/bin/python3 FRONT_BIN=/opt/gr8/bin/g6front-rs
ENV TUNE=speed DFLASH=0 MENTAT=0
ENV TP=4 GROUP=gr8-sage-4x KV_FORMAT=int4 FLASH_CACHE=aligned
ENV DEFAULT_MAX_TOKENS=32768 UNDECLARED_TOOL_CALLS=emit
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/tmp
WORKDIR /opt/gr8
USER 10001:10001
ENTRYPOINT []
