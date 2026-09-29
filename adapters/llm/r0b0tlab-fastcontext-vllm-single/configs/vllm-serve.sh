#!/usr/bin/env bash
# Serve r0b0tlab/FastContext-1.0-4B-RL-NVFP4 (vLLM 0.23+, NVFP4-capable GPU).
set -euo pipefail

MODEL="${FC_NVFP4_MODEL:-r0b0tlab/FastContext-1.0-4B-RL-NVFP4}"
PORT="${FC_NVFP4_PORT:-30000}"

exec vllm serve "$MODEL" \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --gpu-memory-utilization 0.40 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name FastContext-1.0-4B-RL-NVFP4 \
  --host 0.0.0.0 \
  --port "$PORT"