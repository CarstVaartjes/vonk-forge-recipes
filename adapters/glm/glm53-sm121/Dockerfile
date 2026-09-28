FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark" \
      org.opencontainers.image.revision="aed98a13ca75140d2691cc5c651ea5817d9a3e44" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0 AND BSD-3-Clause"

# vLLM v0.30.0 (aarch64, CUDA 13.0) is the first stable release with GLM-5.3-Flash
# support and already ships FlashInfer 0.6.18.post1, which the NoPE MLA path needs
# on GB10, plus NCCL 2.30.7, the version both upstream Spark profiles exercise.
# Pin Ray and assert those versions before applying the consolidated SM121 fix.
RUN pip install --no-cache-dir "ray[default]==2.58.0" \
    && pip show flashinfer-python | grep -q "Version: 0.6.18.post1" \
    && pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7"

COPY glm53-flash_SM121.py /opt/vonk/build/glm53-flash_SM121.py
COPY install-nccl.py verify-runtime.py /opt/vonk/build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY chat_template_mm.jinja /opt/vonk/templates/glm53-chat-template-mm.jinja

RUN python3 /opt/vonk/build/glm53-flash_SM121.py \
    && python3 /opt/vonk/build/install-nccl.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
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
