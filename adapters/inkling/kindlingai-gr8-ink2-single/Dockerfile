# KindlingAI's Inkling-2-Mini exlr8 recipe for one DGX Spark (kindlingai/recipes at commit 13c24ac4, MIT, the kit
# https://github.com/kindlingai/recipes/tree/13c24ac483eaf8b2fcb828a62ae2604bcc038b92/inkling-2-mini/dgx-spark/1x/exlr8-gr8)
# runs gr8's published engine image as it is: gr8 main commit 2c221e98, its from-scratch CUDA engine for Inkling-2
# (g6m4_ink2), the S2 shim, the Rust front end and the loader on nvidia/cuda 13.0.1 (arm64). The kit compiles nothing and
# names this image by tag and digest in its compose.yaml; it runs its own serve.sh, not gr8's image entrypoint. This layer
# adds that serve.sh at the kit's path, the Controller placement wrapper at the platform's launcher path and the
# non-root user. Base: the linux/arm64 image ghcr.io/kindlingai/gr8:sha-2c221e9, by digest.
FROM ghcr.io/kindlingai/gr8@sha256:dfa8eb1a4e967338f3ab087f980479b40587ba6362da61950fff3ddd7d7717ce

LABEL org.opencontainers.image.source="https://github.com/kindlingai/recipes" \
      org.opencontainers.image.revision="13c24ac483eaf8b2fcb828a62ae2604bcc038b92" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY --chmod=0755 serve.sh /recipe/serve.sh
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/bin/g6m4_ink2 && test -x /opt/gr8/bin/g6front-rs && test -f /opt/gr8/frontend/g6s2shim.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The kit's compose environment (PY; its ENGINE_ROOT is serve.sh's own default, /opt/gr8). The kit keeps its working directory in /tmp; the wrapper moves it under
# /outputs/cache/gr8, the platform's writable state. The model is used in place from the read-only /models mounts.
ENV PY=/usr/bin/python3 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/tmp
WORKDIR /opt/gr8
USER 10001:10001
ENTRYPOINT []
