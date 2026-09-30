#!/bin/bash
# Laguna M.1 NVFP4 — dual GB10 TP=2 (Ray). Profiles: firstboot (default) | optimized (L1).
set -e

export CUDA_HOME=/usr/local/cuda
TORCH_LIB=$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)
export LD_LIBRARY_PATH="${TORCH_LIB}:/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

export TORCH_CUDA_ARCH_LIST=12.1a
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_HCA=rocep1s0f0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export TP_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

PORT="${PORT:-30100}"
SERVED_NAME="${SERVED_NAME:-laguna-m1-nvfp4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_BATCHED="${MAX_BATCHED:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
DIST_TIMEOUT="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
EXEC_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-ray}"
BLOCK_SIZE="${KV_BLOCK_SIZE:-256}"

EXTRA=()
COMPILE_ARGS=()
if [[ "${ENFORCE_EAGER:-1}" == "1" || "${ENFORCE_EAGER}" == "true" ]]; then
  EXTRA+=(--enforce-eager)
else
  CG="${CUDAGRAPH_MODE:-PIECEWISE}"
  COMPILE_ARGS=(--compilation-config "{\"cudagraph_mode\":\"${CG}\",\"custom_ops\":[\"all\"]}")
fi

echo "$(date -Is) Laguna TP=2 backend=$EXEC_BACKEND load=$LOAD_FORMAT len=$MAX_MODEL_LEN kv=$KV_CACHE_DTYPE util=$GPU_UTIL eager=${ENFORCE_EAGER:-1}"

exec vllm serve /mnt/model \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend "$EXEC_BACKEND" \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds "$DIST_TIMEOUT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --block-size "$BLOCK_SIZE" \
  --load-format "$LOAD_FORMAT" \
  "${EXTRA[@]}" \
  "${COMPILE_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --default-chat-template-kwargs '{"enable_thinking": true}'