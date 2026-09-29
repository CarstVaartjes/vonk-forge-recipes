#!/usr/bin/env bash
# 1M YaRN + DFlash2 n=7 on one DGX Spark (AEON single-Spark recipe + nested rope).
# Drafter config is YaRN-extended too (vLLM does not stretch dflash rope past 262k).
# GMU 0.70. Never above 0.85.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4}"
DRAFT="${DRAFT:-$HOME/models/Qwen3.8-27B-DFlash2}"
# Community prebuild (AEON-7 aeon-vllm-ultimate retag). Identical digest dd201847.
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-qwen38-27b-nvfp4-dflash2-ablit-cyber-unlock-1m-yarn-single-dgxspark:latest}"
NAME="${NAME:-aeon-b5}"
PORT="${PORT:-8000}"
TEMPLATE="${TEMPLATE:-$HERE/chat_template_b5_uncensored.jinja}"
DRAFT_CFG="${DRAFT_CFG:-$HERE/dflash2_config_yarn_1m.json}"
YARN="${YARN:-$HERE/yarn_hf_overrides.json}"
GMU="${GMU:-0.70}"
python3 -c "u=float('$GMU'); assert u<=0.85, 'GMU must be <= 0.85'"

[[ -f "$MODEL/model.safetensors" || -f "$MODEL/model.safetensors.index.json" ]] || { echo "missing uniform NVFP4 $MODEL"; exit 1; }
[[ -f "$DRAFT/config.json" ]] || { echo "missing DFlash2 $DRAFT"; exit 1; }
[[ -f "$TEMPLATE" && -f "$DRAFT_CFG" && -f "$YARN" ]] || { echo "missing yarn overlay files in $HERE"; exit 1; }
python3 -c "import json; json.load(open('$YARN'))"

HF_OVR=$(python3 -c "import json; print(json.dumps(json.load(open('$YARN')), separators=(',',':')))")

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --gpus all --network host --ipc host \
  --name "$NAME" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "$MODEL:/model:ro" \
  -v "$DRAFT:/drafter:ro" \
  -v "$TEMPLATE:/model/chat_template.jinja:ro" \
  -v "$DRAFT_CFG:/drafter/config.json:ro" \
  --entrypoint vllm "$IMAGE" serve /model \
  --served-model-name aeon \
  --host 0.0.0.0 --port "$PORT" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN \
  --max-model-len 1048576 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$GMU" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --hf-overrides "$HF_OVR" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN"}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.05}' \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --trust-remote-code
echo "launched $NAME 1M-YaRN/seqs2/DFlash2-n7 gmu=$GMU"
docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}} {{.Image}}'
