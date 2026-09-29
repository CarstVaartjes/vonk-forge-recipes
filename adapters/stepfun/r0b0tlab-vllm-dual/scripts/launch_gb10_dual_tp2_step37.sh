#!/usr/bin/env bash
# TP=2 launch candidate for Step 3.7 Flash NVFP4 on dual GB10.
# This is an operational smoke script, not the final published launch script.
set -euo pipefail
cd "$(dirname "$0")/.."
. docker/ray_cluster.env

MODEL=${MODEL:-/home/r0b0tdgx/models/llm/nvfp4/stepfun-ai/Step-3.7-Flash-NVFP4}
PY=${PY:-/home/r0b0tdgx/vllm-gb10/.venv/bin/python}
PORT=${PORT:-30000}
GPU_UTIL=${GPU_UTIL:-0.70}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_SEQS=${MAX_SEQS:-1}
BATCHED=${BATCHED:-8192}
LOG=${LOG:-evidence/runtime/vllm-tp2-launch.log}

mkdir -p "$(dirname "$LOG")"

export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export TP_SOCKET_IFNAME=enp1s0f0np0
export VLLM_HOST_IP=192.168.100.10
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

echo "=== Step3.7 NVFP4 TP=2 launch candidate ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU util: $GPU_UTIL"
echo "Max model len: $MAX_MODEL_LEN"
echo "Log: $LOG"

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --disable-cascade-attn \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$BATCHED" \
  --max-num-seqs "$MAX_SEQS" \
  --reasoning-parser step3p5 \
  --tool-call-parser step3p5 \
  --enable-auto-tool-choice \
  --async-scheduling \
  2>&1 | tee "$LOG"
