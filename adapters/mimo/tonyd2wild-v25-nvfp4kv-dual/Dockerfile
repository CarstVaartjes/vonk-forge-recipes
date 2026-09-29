FROM ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv@sha256:219043cf57a42499ccc514f4de5b96f9e8e401fee47e435ba96ce06c3171f1c8

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark" \
      org.opencontainers.image.revision="6dae0a60dd3f90580029c2dd981a0cd5de49fc4e" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# The published image is a vLLM 0.21.1rc1.dev85 dev build without the MiMo V2.5
# mods; upstream applies them with `docker cp` and `docker exec` after starting
# the container. Apply the same four vendored mods, in upstream's order, at build
# time instead. The two other upstream mods (drop-caches, ray-keep-node-nccl-hca)
# are host and orchestration helpers and are left out. fix-mimo-v2-vllm downloads
# vLLM PR 41797 (merged 2026-06-11) when the image lacks the DiffKV backend.
COPY mods /opt/vonk/build/mods
COPY verify-runtime.py /opt/vonk/verify-runtime.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN set -eu; for mod in fix-prometheus-instrumentator-router fix-mimo-v2-vllm fix-modelopt-mixed-mxfp8 nvfp4-kv-diffkv; do bash "/opt/vonk/build/mods/$mod/run.sh"; done \
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
    TORCH_CUDA_ARCH_LIST=12.1a \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
