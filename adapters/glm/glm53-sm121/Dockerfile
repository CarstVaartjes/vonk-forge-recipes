# MiaAI-Lab's published GLM-5.3-Flash NVFP4 two-Spark serving image (kit
# https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark at aed98a13ca75140d2691cc5c651ea5817d9a3e44):
# the kit's kernel layer glm53-flash-sm121:v8 (SM90 sparse-MLA and FlashInfer FA2 selected for GB10, FlashInfer 0.6.18
# nightly, NCCL 2.30.7, cutlass-dsl 4.6.2) plus its mm-ray-v1 serving layer (Ray 2.58.0, multimodal defaults). The kit's
# README gives it as ghcr.io/miaai-lab/glm53-flash-spark:mm-ray-v1; this recipe runs that image by digest and does not
# rebuild it from the kit's sources. This layer only adds the Controller placement wrapper at the platform's launcher
# path, the multimodal chat template, the image's NCCL 2.30.7 at the path the wrapper preloads, the licence and the
# non-root user. Base: the linux/arm64 image ghcr.io/miaai-lab/glm53-flash-spark, by digest.
FROM ghcr.io/miaai-lab/glm53-flash-spark@sha256:376f11dfd61abb06731d2d542b53c8c2954ffbcf1846ccd3af564d507ec5436f

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark" \
      org.opencontainers.image.revision="aed98a13ca75140d2691cc5c651ea5817d9a3e44" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0 AND BSD-3-Clause" \
      ai.vonkforge.runtime-interface="v1"

COPY LICENSE /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY chat_template_mm.jinja /opt/vonk/templates/glm53-chat-template-mm.jinja
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py

# The kit's image carries the pip nvidia-nccl-cu13 2.30.7 (its build asserts the version); the wrapper preloads it
# from the platform path, so it is copied there instead of downloaded again.
RUN set -eu; \
    nccl="$(find /usr/local/lib -path '*/nvidia/nccl/lib/libnccl.so.2' | head -n 1)"; \
    test -n "$nccl"; \
    pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7"; \
    install -D -m 0555 "$nccl" /opt/vonk/lib/libnccl.so.2; \
    test -x /usr/local/bin/vllm; \
    python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /inputs \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    VLLM_NCCL_SO_PATH=/opt/vonk/lib/libnccl.so.2 \
    LD_PRELOAD=/opt/vonk/lib/libnccl.so.2 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    RAY_memory_usage_threshold=0.99 \
    RAY_memory_monitor_refresh_ms=0 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
