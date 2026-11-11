FROM docker.io/eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595

LABEL org.opencontainers.image.source="https://github.com/Reederey87/glm53-flash-fp4-b12x-2x-dgx-spark" \
      org.opencontainers.image.revision="29d0989cf60a82f653c33a25c871d00cba9ae1fe" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT"

# The kit runs eugr's published vllm-node-b12x image: its launch.sh IMAGE_REF default is
# eugr/spark-vllm-b12x:nightly-20261015 (FlashInfer 0.7.1 b12x, built 2026-10-15), locked here
# by its linux/arm64 digest. The creator's own deployment is a local retag of an unpublished
# build of that lane, so the published digest is the nearest runnable image (see the PR).
# Upstream starts it with `vllm serve`; the wrapper is the Vonk entrypoint.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /opt/vonk/bin/vllm \
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
