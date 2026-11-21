# KindlingAI's GLM-5.3 exlr8 recipe for six DGX Sparks (kindlingai/glm-5.3-exlr8-6x-spark at commit d2270fc4, MIT) runs gr8's published
# engine image as it is, with the recipe's settings in the environment: gr8 v1.1.0 (commit cdd1e093), its from-scratch
# CUDA engine for every TP width (g6m4-int4 here), the Rust front end, the loader and the launcher on nvidia/cuda 13.0.1
# (arm64). The kit compiles nothing and names this image by tag and digest in its compose.yaml. This layer only adds
# the Controller placement wrapper at the platform's launcher path, the licence text and the non-root user.
# Base: the linux/arm64 image ghcr.io/kindlingai/gr8:v1.1.0, by digest.
FROM ghcr.io/kindlingai/gr8@sha256:b22eedbd7902f65dbf2090897d75f5d92276cbf78a16bcde1f958aa56a306cf7

LABEL org.opencontainers.image.source="https://github.com/kindlingai/glm-5.3-exlr8-6x-spark" \
      org.opencontainers.image.revision="d2270fc41e7050d73cf7b17b0e7661a34ce6cc5b" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/docker/entrypoint.sh && test -x /opt/gr8/bin/g6m4-int4 \
    && test -x /opt/gr8/bin/g6load && test -x /opt/gr8/bin/g6front-rs \
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
