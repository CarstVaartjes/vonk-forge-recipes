FROM ghcr.io/drowzeys/eugr-gb10-nvfp4kv@sha256:d9010635c22a34694000447f96bf12622210494578be670b2a9e953ca04b29af

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark" \
      org.opencontainers.image.revision="31f6197a4c59049431feda7b291ada63d92c22a5" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# The pinned image is upstream's published :a4q2 stack (eugr spark-vllm-b12x
# nightly-20260813 + NVFP4-KV FA2 back-port + A4Q fp4-QK overlay). recipe/ is
# upstream's build source for that image, carried byte-identically for
# provenance; nothing in it is rebuilt here.
COPY recipe /opt/vonk/upstream-recipe

RUN install --directory --mode=0755 /opt/vonk/bin \
    && ln -s "$(command -v vllm)" /opt/vonk/bin/vllm \
    && test -x /opt/vonk/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch-extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
