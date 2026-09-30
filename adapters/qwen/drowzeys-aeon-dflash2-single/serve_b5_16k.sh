#!/usr/bin/env bash
# 16k / seqs 64 rollback (speed/conc seat). Default live seat is serve_b5_on_4.sh (1M YaRN).
# GMU 0.75. Never above 0.85.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4}"
DRAFT="${DRAFT:-$HOME/models/Qwen3.8-27B-DFlash2}"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-qwen38-27b-nvfp4-dflash2-ablit-cyber-unlock-1m-yarn-single-dgxspark:latest}"
NAME="${NAME:-aeon-b5}"
PORT="${PORT:-8000}"
TEMPLATE="${TEMPLATE:-$HERE/chat_template_b5_uncensored.jinja}"
GMU="${GMU:-0.75}"
python3 -c "u=float('$GMU'); assert u<=0.85, 'GMU must be <= 0.85'"

[[ -f "$MODEL/model.safetensors" || -f "$MODEL/model.safetensors.index.json" ]] || { echo "missing uniform NVFP4 $MODEL"; exit 1; }
[[ -f "$DRAFT/config.json" ]] || { echo "missing DFlash2 $DRAFT"; exit 1; }
[[ -f "$TEMPLATE" ]] || { echo "missing $TEMPLATE"; exit 1; }

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --gpus all --network host --ipc host \
  --name "$NAME" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 \
  -v "$MODEL:/model:ro" \
  -v "$DRAFT:/drafter:ro" \
  -v "$TEMPLATE:/model/chat_template.jinja:ro" \
  --entrypoint vllm "$IMAGE" serve /model \
  --served-model-name aeon \
  --host 0.0.0.0 --port "$PORT" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization "$GMU" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN"}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.05}' \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --trust-remote-code
echo "launched $NAME B5 16k/seqs64/DFlash2-n7 image=$IMAGE port=$PORT"
docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}} {{.Image}}'
