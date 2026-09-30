FROM ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c@sha256:f3b0577bfec41ac2cece2cbe0c4a9be934d9d4f18c50bbcdd602848da2499fb8

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-vLLm-0.24.0-Optimized-DeepSeekV4-Flash-DSpark-NVFP4-KV-1.5M-CTX-3M-Pool-C-12-on-2-DGX-Spark" \
      org.opencontainers.image.revision="9d7346a1c540d849520d25a5a2a8b5129ded2901" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: drowzeys' published stage-c runtime (the image upstream's launcher
# calls vllm-dspark-runtime:dspark-nvfp4-stage-c; arm64, CUDA 13, conda env at
# /opt/env). Its /opt/env/bin/vllm is the engine; this layer adds the placement
# wrapper, the licences, upstream's ops/ launchers (provenance only) and the
# non-root user. Cache locations baked into the parent point at /cache/huggingface
# and are redirected to the platform-owned /outputs/cache.
COPY LICENSE NOTICE /opt/vonk/licenses/
COPY ops /opt/vonk/upstream-ops
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /opt/env/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state \
    HF_HOME=/outputs/cache/huggingface \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm-cache \
    TRITON_CACHE_DIR=/outputs/cache/triton-cache \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor-cache \
    TILELANG_CACHE_DIR=/outputs/cache/tilelang \
    DG_JIT_CACHE_DIR=/outputs/cache/deepgemm-cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache/flashinfer

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
