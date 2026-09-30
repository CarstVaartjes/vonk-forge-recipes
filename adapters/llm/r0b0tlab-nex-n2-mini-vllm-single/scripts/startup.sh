#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/mnt/model}"

# Download model from HF if not present
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "[startup] Model not found at $MODEL_DIR"
    echo "[startup] Downloading r0b0tlab/nex-n2-mini-nvfp4 from HuggingFace..."
    mkdir -p "$MODEL_DIR"
    hf download r0b0tlab/nex-n2-mini-nvfp4 \
        --local-dir "$MODEL_DIR" \
        --token "${HF_TOKEN}" \
        --exclude "*.md"
    echo "[startup] Download complete."
fi

echo "[startup] Starting vLLM server on port ${PORT:-8000}..."
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port "${PORT:-8000}" \
    --quantization modelopt_fp4 \
    --trust-remote-code \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --moe-backend flashinfer_cutlass \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --chat-template "$MODEL_DIR/chat_template.jinja" \
    ${VLLM_EXTRA_ARGS:-}
