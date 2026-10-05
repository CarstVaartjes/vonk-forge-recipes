FROM docker.io/eugr/spark-vllm-b12x@sha256:69f91c8fb0414126f0114eb8ab1cf9ea936dd0a1e7d5763cf80766c9403831d1

LABEL org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.revision="bb6ee761643f45eb84b14f7e39c41d7538d53890" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# eugr's published vllm-node-b12x image (nightly-20260929, `--exp-b12x` lane). Its
# /workspace/build-metadata.yaml records build_script_commit 42f62e34 (the revision above), the
# local-inference-lab/vllm dev/karmic-kraken vLLM fork and the lukealonso b12x kernels.
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
