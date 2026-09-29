#!/usr/bin/env bash
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121:cu130-sm121-arm64-dda4668b}
MODEL_DIR=${MODEL_DIR:-/home/r0b0tdgx/models/llm/nvfp4/poolside/Laguna-M.1-NVFP4}
PORT=${PORT:-30100}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
GPU_UTIL=${GPU_UTIL:-0.85}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-nvfp4}
NAME=${NAME:-laguna-m1-vllm}
SERVED_NAME=${SERVED_NAME:-laguna-m1-nvfp4}
MAX_BATCHED="${MAX_BATCHED:-4096}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

EXTRA_VLLM_ARGS=()
if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
  EXTRA_VLLM_ARGS+=(--enforce-eager)
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing model at $MODEL_DIR — run: hf download poolside/Laguna-M.1-NVFP4 --local-dir $MODEL_DIR"
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --gpus all --ipc=host --network=host \
  --shm-size=16g \
  -v "$MODEL_DIR:/model:ro" \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  "$IMAGE" \
  vllm serve /model \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --load-format fastsafetensors \
    "${EXTRA_VLLM_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser poolside_v1 \
    --reasoning-parser poolside_v1 \
    --default-chat-template-kwargs '{"enable_thinking": true}'

echo "Started $NAME on :$PORT (KV=$KV_CACHE_DTYPE, max_len=$MAX_MODEL_LEN)"
echo "Logs: docker logs -f $NAME"