FROM ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv@sha256:219043cf57a42499ccc514f4de5b96f9e8e401fee47e435ba96ce06c3171f1c8

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark" \
      org.opencontainers.image.revision="8024c85b5e5fc5c9edd9db7b40f607683372cb0c" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# The published image is a vLLM 0.21.1rc1.dev85 dev build without the MiMo V2.5
# mods; upstream applies them with `docker cp` and `docker exec` after starting the
# container. The TP3 repository names the mod list but vendors only nvfp4-kv-diffkv
# (recipe/mods, same repository as the recipe). The three fix-* mods come from
# tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark@6dae0a60 and the TP3 virtual-heads
# mod from tonyd2wild/MiMo-V2.5-Omni-3x-DGX-Spark-TP-3-MTP@12c731f5 (see
# NOTICE.upstream); they are applied here at build time in the recipe's order.
# drop-caches and ray-keep-node-nccl-hca are host and orchestration helpers and are
# left out. fix-mimo-v2-vllm downloads vLLM PR 41797 (merged 2026-06-11) when the
# image lacks the DiffKV backend.
COPY recipe/mods/nvfp4-kv-diffkv /opt/vonk/build/mods/nvfp4-kv-diffkv
COPY mods-external/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark@6dae0a60/ /opt/vonk/build/mods/
COPY mods-external/MiMo-V2.5-Omni-3x-DGX-Spark-TP-3-MTP@12c731f5/mimo-v2-tp3-virtual-heads /opt/vonk/build/mods/mimo-v2-tp3-virtual-heads
COPY verify-runtime.py /opt/vonk/verify-runtime.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN set -eu; for mod in fix-prometheus-instrumentator-router fix-mimo-v2-vllm fix-modelopt-mixed-mxfp8 nvfp4-kv-diffkv mimo-v2-tp3-virtual-heads; do bash "/opt/vonk/build/mods/$mod/run.sh"; done \
    && rm -rf /opt/vonk/build \
    && python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    RAY_memory_monitor_refresh_ms=0 \
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=0 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
