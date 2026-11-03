FROM docker.io/eugr/spark-vllm-b12x@sha256:19ab778dd57f019c44caf5df91a671cf688923d1ca9fa8ce92be0c9acb46d215

LABEL org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.revision="b7f6b4cd4d85c655c5102eeb579f6de6a5bfe3d3" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# eugr's published vllm-node-b12x image (nightly-20261102, `--exp-b12x` lane), built from the
# revision above; the digest is what runs.
# Upstream recipes start it with `vllm serve`; the wrapper is the Vonk entrypoint.
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
