FROM docker.io/eugr/spark-vllm@sha256:5e59cb7efe08a99b023925a185c971c5a31868d8261b755c33d9911c3b9389aa

LABEL org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.revision="bc224d8126036675c6a84b698ed40919027782d2" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# eugr's published vllm-node image (docker.io/eugr/spark-vllm:latest at this digest). Its
# /workspace/build-metadata.yaml records build_script_commit 73a48892 (the revision above), built from
# the repository's default Dockerfile (upstream's local `vllm-node` build) with the stock build arguments.
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
