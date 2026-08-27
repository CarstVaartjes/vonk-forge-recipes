FROM docker.io/vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark" \
      org.opencontainers.image.revision="aed98a13ca75140d2691cc5c651ea5817d9a3e44" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0 AND BSD-3-Clause"

# GLM 5.3's NoPE MLA path needs FlashInfer 0.6.18 on GB10. The nightly changes
# NCCL and Cutlass transitively, so restore the versions physically exercised
# by both upstream Spark profiles before applying their consolidated SM121 fix.
RUN pip install -q --pre \
      flashinfer-python==0.6.18.dev20260819 \
      flashinfer-cubin==0.6.18.dev20260819 \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
    && pip uninstall -q -y flashinfer-jit-cache \
    && pip install -q nvidia-nccl-cu13==2.30.7 nvidia-cutlass-dsl==4.6.2 \
    && pip install --no-cache-dir "ray[default]==2.58.0" \
    && pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7" \
    && pip show nvidia-cutlass-dsl | grep -q "Version: 4.6.2"

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
