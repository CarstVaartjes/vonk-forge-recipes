FROM ghcr.io/drowzeys/vllm-024-puzzle-a4q@sha256:a859a90363d1fee9a4ea37ce507b082fa353e6ac0fb2ddab2fc3295dbd37fe8d

LABEL org.opencontainers.image.source="https://github.com/drowzeys/vLLM-0.24-optimized-NVIDIA-Nemotron-Lab-Puzzle-75B-A9B-A4Q-MTP3-NVFP4-KV-2.7M-Pool-Single-DGX-Spark" \
      org.opencontainers.image.revision="c8432f7909eaa9128f8ad15d951a9e49625e7e88" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# Upstream's published :fi614 image (vLLM 0.24 aeon lineage + jethac FlashInfer
# 0.6.13 with the A4Q wiring). scripts/ is upstream's launcher set, carried
# byte-identically for provenance; the recipe reproduces puzzle-nvfp4-serve.sh.
COPY scripts /opt/vonk/upstream-scripts
COPY LICENSE GB10_SPIN_WAIT_PATCH.md /opt/vonk/

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
