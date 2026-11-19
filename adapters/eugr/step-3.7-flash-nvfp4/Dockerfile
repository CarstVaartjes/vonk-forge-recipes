FROM docker.io/eugr/spark-vllm@sha256:a9070bd0af702459ed8af8da88456be80853e66d6deb908b8fb08c4bca0d546c

LABEL org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.revision="ce3257d06ebb8e3552683d59c553bfc64f43d9e7" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# The published docker.io/eugr/spark-vllm image, built from the revision above; the digest is what runs.
# Upstream recipes start it with `vllm serve`; the wrapper is the Vonk entrypoint.
# eugr applies mods/* with `docker cp` and `docker exec` at container start. Apply the same
# mods at build time instead, in the recipe's order: step-3.7-flash.
# No GPU is needed at build time. On this image the step-3.7-flash mod finds Step-3.7-Flash support already present and skips its patch.
COPY mods /opt/vonk/build/mods
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN set -eu; \
    for mod in step-3.7-flash; do \
      (cd "/opt/vonk/build/mods/$mod" && bash ./run.sh) || exit 1; \
    done \
    && cd / && rm -rf /opt/vonk/build \
    && test -x /opt/vonk/bin/vllm \
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
