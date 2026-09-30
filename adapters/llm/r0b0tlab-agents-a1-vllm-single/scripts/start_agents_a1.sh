#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "audit" ]]; then
  exec /usr/bin/python3 /usr/local/bin/audit_runtime.py
fi

MODEL_PATH="${MODEL_ID:-/models/Agents-A1-NVFP4}"
if [[ ! -e "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_ID path does not exist inside container: $MODEL_PATH" >&2
  echo "Mount the quantization with: -v /path/to/Agents-A1-NVFP4:/models/Agents-A1-NVFP4:ro" >&2
  exit 2
fi

/usr/bin/python3 /usr/local/bin/audit_runtime.py

SPEC_ARGS=()
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  SPEC_ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

exec /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --model "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-agents-a1-nvfp4}" \
  --quantization "${QUANTIZATION:-modelopt}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
  --moe-backend "${MOE_BACKEND:-flashinfer_cutlass}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --trust-remote-code \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  "${SPEC_ARGS[@]}" \
  ${EXTRA_ARGS:-}
