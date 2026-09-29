#!/bin/bash
# run-dsv4-gb10.sh — launch DeepSeek-V4-Flash on a 2-node GB10 cluster using the
# prebuilt GHCR image. Run on the HEAD node (node0). Adjust the vars below to
# your cluster. Requires: model weights present at $MODEL_DIR on BOTH nodes,
# docker with the nvidia runtime on both, passwordless ssh head->worker.
set -euo pipefail

# ---- cluster config (EDIT THESE) ----
IMAGE="ghcr.io/r0b0tlab/vllm-dsv4-flash-gb10:cu130-sm121-arm64-dda4668b"
HEAD_IP="192.168.100.10"
WORKER_IP="192.168.100.11"
ETH_IF="enp1s0f0np0"        # control-plane interface (both nodes)
IB_HCA="rocep1s0f0"         # RoCE HCA
MODEL_DIR="/home/r0b0tdgx/models/llm/source/deepseek-ai/DeepSeek-V4-Flash"
NAME="vllm_ds4"

# ---- profile (pick one) ----
# Latency (best single-stream):   200000 / 2
# Throughput (best aggregate):    65536 / 16
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_BATCHED="${MAX_BATCHED:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MASTER_PORT="29501"

COMMON_ENV=(
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$IB_HCA
  -e NCCL_SOCKET_IFNAME=$ETH_IF -e GLOO_SOCKET_IFNAME=$ETH_IF -e TP_SOCKET_IFNAME=$ETH_IF
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e CUDA_HOME=/usr/local/cuda -e TORCH_CUDA_ARCH_LIST=12.1a -e VLLM_TRITON_MLA_SPARSE=1
)

VLLM_ARGS=(
  serve /mnt/model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8000
  --trust-remote-code --tensor-parallel-size 2 --enable-expert-parallel
  --kv-cache-dtype fp8 --block-size 256 --enable-prefix-caching
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED" --gpu-memory-utilization "$GPU_UTIL"
  --distributed-executor-backend mp
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --speculative-config '{"model":"/mnt/model","num_speculative_tokens":2,"method":"deepseek_mtp"}'
  --tokenizer-mode deepseek_v4 --load-format safetensors --nnodes 2
)

LD="export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\$LD_LIBRARY_PATH;"

echo "Cleaning any prior containers (clears port $MASTER_PORT EADDRINUSE)..."
docker rm -f $NAME 2>/dev/null || true
ssh "$WORKER_IP" "docker rm -f $NAME 2>/dev/null || true"

echo "Starting WORKER (node1, rank 1) first..."
ssh "$WORKER_IP" docker run -d --name $NAME --gpus all --ipc=host --network host \
  -v "$MODEL_DIR":/mnt/model:ro "${COMMON_ENV[@]}" --entrypoint bash "$IMAGE" \
  -c "'$LD vllm ${VLLM_ARGS[*]} --node-rank 1 --master-addr $HEAD_IP --master-port $MASTER_PORT --headless'"

sleep 8
echo "Starting HEAD (node0, rank 0)..."
docker run -d --name $NAME --gpus all --ipc=host --network host \
  -v "$MODEL_DIR":/mnt/model:ro "${COMMON_ENV[@]}" --entrypoint bash "$IMAGE" \
  -c "$LD vllm ${VLLM_ARGS[*]} --node-rank 0 --master-addr $HEAD_IP --master-port $MASTER_PORT"

echo "Launched. Tail logs: docker logs -f $NAME"
echo "Cold start ~6 min. Health: curl http://localhost:8000/health"
