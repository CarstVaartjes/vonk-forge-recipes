# KindlingAI's GLM-5.3-Flash exlr8 gr8 recipe for one DGX Spark (kindlingai/recipes at commit 13c24ac4, MIT, directory
# glm-5.3-flash/dgx-spark/1x/exlr8-gr8), on gr8's published engine image gr8 v1.2.0 (commit ab2eb95f): its from-scratch CUDA engine
# (g6m4_flash-int4), the Rust front end, the loader and the launcher on nvidia/cuda 13.0.1 (arm64). The kit has no
# Dockerfile and names this image by tag and digest in its compose.yaml; this layer is its runtime stage: the kit's
# entrypoint, knobs, launcher shim and prompt-cache file at the kit's paths, plus the Controller placement wrapper at the
# platform's launcher path, the licence text and the non-root user. Nothing compiles.
# Kit: https://github.com/kindlingai/recipes/tree/13c24ac483eaf8b2fcb828a62ae2604bcc038b92/glm-5.3-flash/dgx-spark/1x/exlr8-gr8
# Base: the linux/arm64 image ghcr.io/kindlingai/gr8:v1.2.0, by digest.
FROM ghcr.io/kindlingai/gr8@sha256:3de05aab8cda21fd5d6b4af89ccdba6ff5abdbd704729601fd348fa638ade9fd

LABEL org.opencontainers.image.source="https://github.com/kindlingai/recipes" \
      org.opencontainers.image.revision="13c24ac483eaf8b2fcb828a62ae2604bcc038b92" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

USER 0:0
COPY licenses /opt/vonk/licenses
COPY kv-store.yaml /etc/gr8/kv-store.yaml
COPY --chmod=0755 entrypoint.sh /opt/glm53f-1x/entrypoint.sh
COPY knobs.sh /opt/glm53f-1x/knobs.sh
COPY --chmod=0755 bin/gr8 /opt/glm53f-1x/bin/gr8
COPY --chmod=0755 gr8-serve /opt/vonk/bin/gr8-serve
RUN test -x /opt/gr8/gr8 && test -x /opt/gr8/docker/entrypoint.sh && test -x /opt/gr8/bin/g6m4_flash-int4 \
    && test -x /opt/gr8/bin/g6load && test -x /opt/gr8/bin/g6front-rs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The kit's container paths are in compose.yaml's environment (PY, FRONT_BIN: gr8's image sets the rest) and its settings
# in .env.example, which the recipe's environment carries. gr8 keeps its working files in /var/lib/gr8 volumes; the
# wrapper moves them under /outputs/cache/gr8, the platform's writable state. The model is used in place from the
# read-only /models mounts.
ENV PATH=/opt/glm53f-1x/bin:/opt/gr8/bin:$PATH
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/tmp
WORKDIR /opt/gr8
USER 10001:10001
ENTRYPOINT []
