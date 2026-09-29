#!/usr/bin/env bash
# Static provenance: r0b0tlab.hy3.w4a4.v3.v025.profile.v1
set -euo pipefail
RANK=${1:?rank}; RUN_ID=${2:?run id}; PROFILE=${3:?base|mtp}; GPU_UTIL=${4:?util}
[[ "$RANK" == 0 || "$RANK" == 1 ]]
[[ "$PROFILE" == base || "$PROFILE" == mtp ]]
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_.-]+$ ]]
# GPU sync checking is a diagnostic, not a production launch default.  In
# `error` mode vLLM intentionally aborts MTP's rejection-sampler bookkeeping
# when it copies sampled IDs to CPU.  Keep production MTP unset; use `warn` or
# `error` only for an explicitly labelled diagnostic replay.
GPU_SYNC_CHECK_MODE=${GPU_SYNC_CHECK_MODE:-}
[[ -z "$GPU_SYNC_CHECK_MODE" || "$GPU_SYNC_CHECK_MODE" == warn || "$GPU_SYNC_CHECK_MODE" == error ]]
SYNC_ENV=()
[[ -n "$GPU_SYNC_CHECK_MODE" ]] && SYNC_ENV=(-e "VLLM_GPU_SYNC_CHECK=$GPU_SYNC_CHECK_MODE")
IMAGE='ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121:v0.25.0-cu130-sm121-arm64-702f4814-r2'
MODEL='/opt/hy3/models/Hy3-NVFP4-w4a4-v3'
NAME="hy3-v025-${RUN_ID}-r${RANK}"
if [[ "$RANK" == 0 ]]; then HCA='rocep1s0f1,roceP2p1s0f1'; else HCA='rocep1s0f0,roceP2p1s0f0'; fi
ARGS=(serve /models/Hy3-NVFP4-w4a4-v3
  --quantization modelopt_mixed --served-model-name hy3-w4a4-v3
  --host 0.0.0.0 --port 8004 --tensor-parallel-size 2 --nnodes 2
  --node-rank "$RANK" --master-addr hy3-head.local --trust-remote-code
  --kv-cache-dtype fp8 --gpu-memory-utilization "$GPU_UTIL"
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-1536M}"
  --moe-backend flashinfer_cutlass --max-model-len "${MAX_MODEL_LEN:-2048}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
  --max-num-seqs "${MAX_NUM_SEQS:-1}" --no-async-scheduling
  --no-enable-prefix-caching --optimization-level 0
  --no-enable-flashinfer-autotune --enforce-eager)
if [[ "$PROFILE" == mtp ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-1}}")
fi
[[ "$RANK" == 1 ]] && ARGS+=(--headless)
exec docker run --name "$NAME" --entrypoint vllm --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband/rdma_cm --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 --device /dev/infiniband/uverbs2 \
  --device /dev/infiniband/uverbs3 \
  -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 \
  -e NCCL_IB_HCA="$HCA" -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_IB_MERGE_NICS=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_GID_INDEX=3 -e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2 \
  -e NCCL_DEBUG=INFO -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e OMP_NUM_THREADS=1 \
  -e PYTHONFAULTHANDLER=1 "${SYNC_ENV[@]}" \
  -v "$MODEL":/models/Hy3-NVFP4-w4a4-v3:ro \
  -v /opt/hy3/.cache/flashinfer:/root/.cache/flashinfer \
  -v /dev/infiniband:/dev/infiniband "$IMAGE" "${ARGS[@]}"
