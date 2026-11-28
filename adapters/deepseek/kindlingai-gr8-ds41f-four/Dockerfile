# KindlingAI's DeepSeek-V4.1-Flash exlr8 gr8 recipe for four DGX Sparks (kindlingai/recipes at commit 13c24ac4, MIT, directory
# deepseek-v4.1-flash/dgx-spark/4x/exlr8-gr8), on gr8's published engine image gr8 v1.3.1
# (commit 9f4f9c4): its from-scratch CUDA engine for DeepSeek-V4.1-Flash at TP=4 (g6m4_ds41, fp8 KV, built with DS41_MX=48),
# the Rust front end, the loader and the launcher on nvidia/cuda 13.0.1 (arm64). The kit has no Dockerfile and names this
# image by tag and digest in its compose.yaml; this layer is its runtime stage: the kit's knobs at the kit's path, plus
# the Controller placement wrapper at the platform's launcher path, the licence text and the non-root user. Nothing
# compiles. The kit's seccomp-iouring.json is not used: the sandbox keeps the default profile and Engram reads with the
# pread pool.
# Kit: https://github.com/kindlingai/recipes/tree/13c24ac483eaf8b2fcb828a62ae2604bcc038b92/deepseek-v4.1-flash/dgx-spark/4x/exlr8-gr8
# Base: the linux/arm64 image ghcr.io/kindlingai/gr8:v1.3.1, by digest.
FROM ghcr.io/kindlingai/gr8@sha256:f6d09cca202bd56114fe93f8efdadd216a6c9ede079f002a601d692e90241b63

LABEL org.opencontainers.image.source="https://github.com/kindlingai/recipes" \
      org.opencontainers.image.revision="13c24ac483eaf8b2fcb828a62ae2604bcc038b92" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY licenses /opt/vonk/licenses
COPY knobs.sh /opt/ds41-4x/knobs.sh
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/docker/entrypoint.sh && test -x /opt/gr8/bin/g6m4_ds41 \
    && test -x /opt/gr8/bin/g6load && test -x /opt/gr8/bin/g6front-rs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# gr8's image keeps its working files in the /var/lib/gr8/home volume; the wrapper moves them under /outputs/cache/gr8,
# the platform's writable state, and uses the release in place from the read-only /models mounts. The kit's settings are
# in .env.example, which the recipe's environment carries.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/tmp
WORKDIR /opt/gr8
USER 10001:10001
ENTRYPOINT []
