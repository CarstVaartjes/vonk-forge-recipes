#!/usr/bin/env bash
# DeepSeek-V4-Flash @ 1M context — 2x DGX Spark (GB10), TP=2 over RDMA, no Ray.
# Arg1 = NODE_RANK (0 = head, 1 = worker). LAUNCH WORKER FIRST, THEN HEAD.
#
# >>> EDIT THESE for your pair <<<
MASTER_ADDR="${MASTER_ADDR:-169.254.73.22}"   # head node's direct-link (ConnectX) IP
IFNAME="${IFNAME:-enp1s0f0np0}"               # the 200GbE interface carrying the link (ip addr)
IB_HCA="${IB_HCA:-rocep1s0f0}"                # RDMA device for that port (ibv_devices)
#
set -uo pipefail
NODE_RANK="${1:?usage: ds4-1m-launch.sh <0|1>}"
HEADLESS_FLAG=""
[ "$NODE_RANK" = "1" ] && HEADLESS_FLAG="--headless"

docker rm -f vllm_unholy 2>/dev/null || true

docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  --name vllm_unholy \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME="$IFNAME" -e GLOO_SOCKET_IFNAME="$IFNAME" -e TP_SOCKET_IFNAME="$IFNAME" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  aidendle94/sparkrun-vllm-ds4-gb10:production-ready \
  -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash --served-model-name deepseek-v4-flash-spark deepseek-v4-flash --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --kv-cache-dtype fp8 --block-size 256 --max-model-len 1000000 --max-num-seqs 6 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.82 --enable-prefix-caching --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 --enable-flashinfer-autotune --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER_ADDR --master-port 29501 $HEADLESS_FLAG"

echo "launched vllm_unholy node-rank=$NODE_RANK headless='$HEADLESS_FLAG' rc=$?"
sleep 2
docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_unholy || echo "WARN: container not in ps (check docker logs vllm_unholy)"
