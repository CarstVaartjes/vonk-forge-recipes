#!/usr/bin/env bash
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/r0b0tlab/vllm-diffusiongemma-26b-nvfp4-sm121:spark-dgemma-patched}
MODEL_DIR=${MODEL_DIR:-/home/r0b0tdgx/models/llm/nvfp4/nvidia/diffusiongemma-26B-A4B-it-NVFP4}
PORT=${PORT:-30000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_UTIL=${GPU_UTIL:-0.90}
NAME=${NAME:-dgemma-vllm}
# KV cache: model hf_quant_config declares FP8 KV cache; force fp8 to reduce cache footprint and maximize c16 headroom.
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
ENABLE_THINKING=${ENABLE_THINKING:-false}

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --gpus all --ipc=host --network=host \
  -v "$MODEL_DIR:/model:ro" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  "$IMAGE" \
  vllm serve /model \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --load-format fastsafetensors \
    --attention-backend TRITON_ATTN \
    --enable-prefix-caching \
    --diffusion-config '{"canvas_length":256}' \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --override-generation-config '{"max_new_tokens": null}' \
    --chat-template fixed_chat_template.jinja \
    --default-chat-template-kwargs "{\"enable_thinking\":${ENABLE_THINKING}}"

echo "Started $NAME with KV_CACHE_DTYPE=$KV_CACHE_DTYPE. Follow logs: docker logs -f $NAME"
