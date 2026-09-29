#!/bin/bash
set -e
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp1s0f0np0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-enp1s0f0np0}"
MODEL="${MODEL:-stepfun-ai/Step-3.7-Flash-NVFP4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_SEQS="${MAX_SEQS:-5}"
BATCHED_TOKENS="${BATCHED_TOKENS:-32768}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== Step 3.7 Flash NVFP4 SM121 TP=2 ==="
echo "Model: $MODEL | Port: $PORT | GPU: $GPU_UTIL | Ctx: $MAX_MODEL_LEN | KV: $KV_CACHE_DTYPE"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --trust-remote-code --tensor-parallel-size 2 --distributed-executor-backend ray \
  --disable-custom-all-reduce --quantization modelopt --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --disable-cascade-attn --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$BATCHED_TOKENS" \
  --max-num-seqs "$MAX_SEQS" --reasoning-parser step3p5 --tool-call-parser step3p5 \
  --enable-auto-tool-choice --enforce-eager $EXTRA_ARGS
