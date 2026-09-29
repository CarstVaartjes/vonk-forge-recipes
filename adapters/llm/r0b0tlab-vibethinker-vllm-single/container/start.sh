#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-r0b0tlab/VibeThinker-3B-NVFP4}"
MODEL_DIR="${MODEL_DIR:-/mnt/model}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== VibeThinker-3B-NVFP4 Container ==="

# Download model if not present
if [ ! -f "${MODEL_DIR}/model.safetensors" ] && [ ! -f "${MODEL_DIR}/model-00001-of-00002.safetensors" ]; then
    echo "Model not found at ${MODEL_DIR}, downloading from HF..."
    if [ -n "${HF_TOKEN}" ]; then
        hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}" --token "${HF_TOKEN}"
    else
        hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
    fi
    echo "Download complete."
else
    echo "Model found at ${MODEL_DIR}, skipping download."
fi

echo "Starting vLLM on port ${PORT}..."
exec vllm serve "${MODEL_DIR}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --quantization modelopt \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code
