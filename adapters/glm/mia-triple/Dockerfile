FROM ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks@sha256:f8f350d46b33858eda4f0c0a5c39a7f0b27005111d393098baaaacae18c10fdb

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks" \
      org.opencontainers.image.revision="5c85163ccb8d98d395880e71e2dbd03976a3f4ad" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY install-nccl.py verify-runtime.py /opt/vonk/build/

RUN python3 /opt/vonk/build/install-nccl.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && install --directory --owner 10001 --group 10001 /state /outputs \
    && test -x /opt/vonk/bin/vllm \
    && test -r /opt/vonk/lib/libnccl.so.2

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_NCCL_SO_PATH=/opt/vonk/lib/libnccl.so.2 \
    VLLM_MTP_INDEX_SHARE=1 \
    VLLM_GLM_TP_PAD=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    GLM_MOE_LANE_ROWS=1 \
    GLM_NVFP4_LUT256=1 \
    GLM_SHARED_EXPERTS_DEBUG=1 \
    RAY_memory_usage_threshold=0.99 \
    RAY_memory_monitor_refresh_ms=0 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_DEVICE_MAX_CONNECTIONS=32 \
    OMP_NUM_THREADS=16 \
    SAFETENSORS_FAST_GPU=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
