# KindlingAI's published image of its GLM-5.3-Flash exlr8 gr8 recipe for one DGX Spark at commit ad0c7148
# (MIT): gr8's from-scratch CUDA engine (g6m4_flash-int4), its Rust front end, the loader and the launcher on
# nvidia/cuda 13.0.1 (arm64). The recipe repository's image workflow publishes it for every commit (the revision label
# of this image is that commit), and this recipe runs it as published rather than rebuilding it from the gr8 sources.
# This layer only adds the Controller placement wrapper at the platform's launcher path, the licence text and the
# non-root user. Base: the linux/arm64 image ghcr.io/kindlingai/glm-5.3-flash-exlr8-1x-spark, by digest.
FROM ghcr.io/kindlingai/glm-5.3-flash-exlr8-1x-spark@sha256:83035720871697937af33892fb900ddebce3020c6fb444d018c4037b2ea4f3a8

LABEL org.opencontainers.image.source="https://github.com/kindlingai/glm-5.3-flash-exlr8-1x-spark" \
      org.opencontainers.image.revision="ad0c7148e3ea73b40f639a1a891357aff0e3d14b" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/bin/g6m4_flash-int4 && test -x /opt/gr8/bin/g6load \
    && test -x /opt/gr8/bin/g6front-rs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The creator's image keeps its working files in /var/lib/gr8 volumes; the wrapper moves them under /outputs/cache/gr8,
# the platform's writable state. The model is used in place from the read-only /models mounts.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/tmp
WORKDIR /opt/gr8
USER 10001:10001
ENTRYPOINT []
