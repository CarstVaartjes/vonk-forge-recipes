# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:26.01-py3
ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
    VLLM_USE_FLASHINFER_MOE_FP4=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    SAFETENSORS_FAST_GPU=1 \
    OMP_NUM_THREADS=8 \
    VLLM_CACHE_ROOT=/root/.cache/vllm
RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build cmake curl jq && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/minimax-m27
COPY src/vllm /opt/vllm
RUN python3 -m pip install -U pip setuptools wheel packaging && \
    cd /opt/vllm && \
    TORCH_CUDA_ARCH_LIST="12.0" VLLM_TARGET_DEVICE=cuda python3 -m pip install -e . --no-build-isolation
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
