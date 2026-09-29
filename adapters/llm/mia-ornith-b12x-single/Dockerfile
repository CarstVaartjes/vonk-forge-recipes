FROM docker.io/eugr/spark-vllm-b12x@sha256:25fe41c2e85993b4e0534b3c72f68bf327c5d2726fbe1640cf6b220715d3b0e3

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark" \
      org.opencontainers.image.revision="9fd7fe7896230550fad1c773e85ad765292cfde5" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# The base is eugr's nightly-20260815 (upstream names the mutable :latest tag).
# Upstream start.sh bind-mounts two patched b12x files read-only over the
# image's copies to make the b12x W4A16 MoE path CUDA-graph safe with MTP.
# Bake the same exact files in instead; fail the build if the targets moved.
ARG B12X_W4A16=/usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16
RUN test -f "$B12X_W4A16/kernel.py" && test -f "$B12X_W4A16/route_pack.py"
COPY patches/kernel.py ${B12X_W4A16}/kernel.py
COPY patches/route_pack.py ${B12X_W4A16}/route_pack.py

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
