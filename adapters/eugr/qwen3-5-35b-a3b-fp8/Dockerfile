FROM docker.io/eugr/spark-vllm@sha256:a9070bd0af702459ed8af8da88456be80853e66d6deb908b8fb08c4bca0d546c

LABEL org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.revision="ce3257d06ebb8e3552683d59c553bfc64f43d9e7" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# The published docker.io/eugr/spark-vllm image, built from the revision above; the digest is what runs.
# Upstream recipes start it with `vllm serve`; the wrapper is the Vonk entrypoint.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

# Upstream applies mods/* with docker cp/exec at container start; they are baked here at build time,
# in the order the recipe lists them (a mod's own patch steps tolerate non-applicable patches).
COPY mods /opt/vonk/mods
RUN set -e; \
    echo '=== mod fix-qwen3-coder-next'; cd /opt/vonk/mods/fix-qwen3-coder-next; WORKSPACE_DIR=/workspace/vllm PYTHON_ROOT=/usr/local/lib/python3.12/dist-packages bash run.sh; \
    echo '=== mod fix-qwen3.5-chat-template'; cd /opt/vonk/mods/fix-qwen3.5-chat-template; WORKSPACE_DIR=/workspace/vllm PYTHON_ROOT=/usr/local/lib/python3.12/dist-packages bash run.sh; \
    cd /; rm -rf /opt/vonk/mods

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
