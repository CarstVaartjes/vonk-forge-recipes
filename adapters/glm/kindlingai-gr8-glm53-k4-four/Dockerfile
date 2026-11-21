# KindlingAI's published image of its GLM-5.3 exlr8 K4 recipe for four DGX Sparks at commit edde1cc1 (MIT): gr8 v1.1.0
# (commit cdd1e093, the kit's gr8/ submodule) with only the engine this recipe runs (g6m4-int4 at TP=4), the Rust front
# end, the loader and the launcher on nvidia/cuda 13.0.1 (arm64), plus the bit387 width manifest and the recipe's
# settings in the image environment. The kit's CI publishes it for every commit (the revision label of this image is
# that commit), and this recipe runs it as published rather than rebuilding it from the gr8 sources. This layer only
# adds the Controller placement wrapper at the platform's launcher path, the licence text and the non-root user.
# Base: the linux/arm64 image ghcr.io/kindlingai/glm-5.3-exlr8-k4-4x-spark, by digest.
FROM ghcr.io/kindlingai/glm-5.3-exlr8-k4-4x-spark@sha256:593080cb0d4f7c6b815efc4c156b84036fc5c56ae86fbd912a6b221974feb219

LABEL org.opencontainers.image.source="https://github.com/kindlingai/glm-5.3-exlr8-k4-4x-spark" \
      org.opencontainers.image.revision="edde1cc1a8f1ba8b46ec86bd51d509a85a848cf1" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/docker/entrypoint.sh && test -x /opt/gr8/bin/g6m4-int4 \
    && test -x /opt/gr8/bin/g6load && test -x /opt/gr8/bin/g6front-rs && test -f /etc/gr8/exlr8-manifest-bit387.json \
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
