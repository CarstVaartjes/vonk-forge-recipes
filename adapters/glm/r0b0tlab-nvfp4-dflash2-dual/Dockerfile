FROM docker.io/vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/glm53-flash-nvfp4-sm121" \
      org.opencontainers.image.revision="222697317a4a165767ab8ef17b44cf969a5ece09" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# Base: vLLM 0.28.1rc1.dev580 (arm64, CUDA 13.0, FlashInfer 0.6.18). The four
# anchored SM121/GLM patches below are the upstream overlay (container/Dockerfile.overlay
# at the pinned commit); each is idempotent and fails closed on anchor drift.
COPY patches/ /opt/vonk/build/patches/
RUN python3 /opt/vonk/build/patches/0001-sm90-nope-mla-sm121.py \
 && python3 /opt/vonk/build/patches/0002-indexer-persistent-topk-sm121.py \
 && python3 /opt/vonk/build/patches/0003-glm-dflash2-aux-capture.py \
 && python3 /opt/vonk/build/patches/0004-glm-dflash2-drafter-group.py \
 && python3 /opt/vonk/build/patches/verify_overlay.py \
 && rm -rf /opt/vonk/build

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
