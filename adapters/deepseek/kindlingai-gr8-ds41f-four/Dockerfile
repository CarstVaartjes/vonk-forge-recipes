# KindlingAI's published image of its DeepSeek-V4.1-Flash exlr8 gr8 recipe for four DGX Sparks at commit 4038bf81
# (MIT): gr8's from-scratch CUDA engine for DeepSeek-V4.1-Flash at TP=4 (g6m4_ds41, fp8 KV, built with DS41_MX=48), its
# Rust front end, the loader, the launcher and Engram's token map, on nvidia/cuda 13.0.1 (arm64). The recipe repository's
# image workflow publishes it for every commit (the revision label of this image is that commit), and this recipe runs it
# as published rather than rebuilding it from the gr8 sources. This layer only adds the Controller placement wrapper at
# the platform's launcher path, the licence text and the non-root user. Base: the linux/arm64 image
# ghcr.io/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark (tag sha-4038bf8), by digest.
FROM ghcr.io/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark@sha256:c5f655619bcc8e5000f20eaece9cbb992eddbf15b54329dd54b7cd0083565089

LABEL org.opencontainers.image.source="https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark" \
      org.opencontainers.image.revision="4038bf81f08ee049e6f12d51261b3db56a986f95" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/docker/entrypoint.sh && test -x /opt/gr8/bin/g6m4_ds41 \
    && test -x /opt/gr8/bin/g6load && test -x /opt/gr8/bin/g6front-rs \
    && test -f /opt/ds41-4x/knobs.sh && test -s /opt/ds41-4x/ds41.engram.cmap \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The creator's image keeps its working files in the /var/lib/gr8/home volume and the release in /var/lib/gr8/release;
# the wrapper moves the first under /outputs/cache/gr8, the platform's writable state, and uses the release in place
# from the read-only /models mounts.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/tmp
WORKDIR /opt/gr8
USER 10001:10001
ENTRYPOINT []
