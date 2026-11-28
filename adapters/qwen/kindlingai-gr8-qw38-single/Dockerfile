# KindlingAI's Qwen3.8-Flash-Next exlr8 recipe for one DGX Spark (kindlingai/recipes at commit 13c24ac4, MIT, the
# directory https://github.com/kindlingai/recipes/tree/13c24ac483eaf8b2fcb828a62ae2604bcc038b92/qwen3.8-flash-next/dgx-spark/1x/exlr8-gr8)
# runs gr8's published engine image as it is, with the recipe's settings in the environment: gr8 main at fa5d82d3 (image
# tag sha-fa5d82d), its from-scratch CUDA engine for every TP width (g6m4_qw38 here), the Rust front end, the loader and
# the launcher on nvidia/cuda (arm64). The kit compiles nothing and names this image by tag and digest in its
# compose.yaml. This layer only adds the Controller placement wrapper at the platform's launcher path, the licence text
# and the non-root user.
# Base: the linux/arm64 image ghcr.io/kindlingai/gr8:sha-fa5d82d, by digest.
FROM ghcr.io/kindlingai/gr8@sha256:494c44b2315f607a3fabe1fb38cd1b1ddd7074bddfea81a7227451ad207f3c59

LABEL org.opencontainers.image.source="https://github.com/kindlingai/recipes" \
      org.opencontainers.image.revision="13c24ac483eaf8b2fcb828a62ae2604bcc038b92" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/docker/entrypoint.sh && test -x /opt/gr8/bin/g6m4_qw38 \
    && test -x /opt/gr8/bin/g6front-rs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# gr8's image keeps its working files in /var/lib/gr8 volumes; the wrapper moves them under /outputs/cache/gr8, the
# platform's writable state. The model is used in place from the read-only /models mounts.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/tmp
WORKDIR /opt/gr8
USER 10001:10001
ENTRYPOINT []
