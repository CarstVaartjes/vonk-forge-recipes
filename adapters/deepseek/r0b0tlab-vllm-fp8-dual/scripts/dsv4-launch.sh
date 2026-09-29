#!/bin/bash
# DSV4-Flash optimized launch script for launch-cluster.sh.
# CONTRACT: launch-cluster.sh APPENDS --nnodes/--node-rank/--master-addr/
# --master-port to the LAST command here. So this must end in a single
# `exec vllm serve ...` line with NO control flow and NO node/master flags.
set -e

# Pinned-image env fixes: image default env omits CUDA_HOME and torch/lib.
export CUDA_HOME=/usr/local/cuda
TORCH_LIB=$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)
export LD_LIBRARY_PATH="${TORCH_LIB}:/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH}"
# Recipe-validated env (proven SM121 path)
export TORCH_CUDA_ARCH_LIST=12.1a
export VLLM_TRITON_MLA_SPARSE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TILELANG_CLEANUP_TEMP_FILES=1

# Install DeepGEMM (native MXFP4 / FP8 block-scaled tensor-core path)
if [ -d /mnt/deep_gemm ]; then
    git config --global --add safe.directory /mnt/deep_gemm 2>/dev/null || true
    ( cd /mnt/deep_gemm && CUDA_HOME=/usr/local/cuda python3 setup.py install 2>&1 | tail -3 )
fi

# DeepGEMM JIT
export DG_JIT_USE_NVRTC=0
export DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc

# NCCL over RoCE (NOT socket). IB_DISABLE=0 + rocep HCA.
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_HCA=rocep1s0f0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export TP_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_BATCHED="${MAX_BATCHED:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"

echo "$(date -Is) DSV4 optimized: EP on, MTP=2, len=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS RoCE"

# Single command — launcher appends --nnodes/--node-rank/--master-* here.
exec vllm serve /mnt/model \
    --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 --block-size 256 \
    --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --distributed-executor-backend mp \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --speculative-config '{"model":"/mnt/model","num_speculative_tokens":2,"method":"deepseek_mtp"}' \
    --tokenizer-mode deepseek_v4 \
    --load-format safetensors
