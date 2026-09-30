#!/usr/bin/env bash
# Dual GB10 TP=2 serve for Laguna M.1 NVFP4 (required — TP=1 OOM on single Spark).
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-/home/r0b0tdgx/models/llm/nvfp4/poolside/Laguna-M.1-NVFP4}"
SPARK=/home/r0b0tdgx/spark-vllm-docker
NODES="${NODES:-192.168.100.10,192.168.100.11}"
IMAGE="${IMAGE:-ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121:cu130-sm121-arm64-dda4668b}"
NAME="${NAME:-laguna-m1-vllm}"
SSH_KEY="${SSH_KEY:-/home/r0b0tdgx/.ssh/id_ed25519_shared}"
REMOTE=r0b0tdgx@192.168.100.11

[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || { echo "missing model on head"; exit 1; }

echo "Checking model on worker..."
ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" \
  "test -f '$MODEL_DIR/model.safetensors.index.json'" || {
  echo "Worker missing weights — run scripts/sync_weights_to_node2.sh first"
  exit 1
}

export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v ${MODEL_DIR}:/mnt/model:ro \
  -e NCCL_IB_GID_INDEX=3 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PORT=${PORT:-30100} \
  -e KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-nvfp4} \
  -e ENFORCE_EAGER=${ENFORCE_EAGER:-1} \
  -e GPU_UTIL=${GPU_UTIL:-0.82} \
  -e MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096} \
  -e MAX_NUM_SEQS=${MAX_NUM_SEQS:-4} \
  -e LOAD_FORMAT=${LOAD_FORMAT:-safetensors} \
  -e DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND:-ray} \
  -e CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-PIECEWISE} \
  -e KV_BLOCK_SIZE=${KV_BLOCK_SIZE:-256} \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}"
export CONTAINER_NCCL_IB_GID_INDEX=3
export CONTAINER_NCCL_IB_DISABLE=0
export CONTAINER_NCCL_SOCKET_IFNAME=enp1s0f0np0
export CONTAINER_GLOO_SOCKET_IFNAME=enp1s0f0np0
export CONTAINER_TP_SOCKET_IFNAME=enp1s0f0np0
export CONTAINER_NCCL_IB_HCA=rocep1s0f0

export CONTAINER_NAME="$NAME"

cd "$SPARK"
./launch-cluster.sh stop --name "$NAME" 2>/dev/null || true
sleep 3

exec ./launch-cluster.sh \
  -n "$NODES" \
  -t "$IMAGE" \
  --name "$NAME" \
  --eth-if enp1s0f0np0 \
  --ib-if rocep1s0f0 \
  --launch-script examples/laguna-m1-launch.sh \
  -d