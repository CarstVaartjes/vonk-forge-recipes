#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. docker/ray_cluster.env

MODEL=${MODEL:-/home/r0b0tdgx/models/llm/nvfp4/stepfun-ai/Step-3.7-Flash-NVFP4}
PY=${PY:-/home/r0b0tdgx/vllm-gb10/.venv/bin/python}
PORT=${PORT:-30000}
GPU_UTIL=${GPU_UTIL:-0.70}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
LOG=${LOG:-evidence/runtime/single-node-smoke.log}

if [ ! -f "$MODEL/config.json" ]; then
  echo "Model metadata missing at $MODEL/config.json"
  exit 1
fi

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp1s0f0np0}"

echo "=== Step3.7 NVFP4 single-node smoke ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU util: $GPU_UTIL"
echo "Max model len: $MAX_MODEL_LEN"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --disable-cascade-attn \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --reasoning-parser step3p5 \
  --tool-call-parser step3p5 \
  --enable-auto-tool-choice \
  --async-scheduling \
  2>&1 | tee "$LOG"
