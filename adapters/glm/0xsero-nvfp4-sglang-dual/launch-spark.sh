#!/bin/bash

set -euo pipefail

required=(NNODES NODE_RANK DIST_INIT_ADDR TP_SIZE EP_SIZE CONTEXT_LENGTH MAX_RUNNING_REQUESTS MEM_FRACTION_STATIC CUDA_GRAPH_MAX_BS CHUNKED_PREFILL_SIZE PORT)
for name in "${required[@]}"; do
  if [[ -z ${!name:-} ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be in [0, NNODES)" >&2
  exit 2
fi

if (( TP_SIZE != NNODES || EP_SIZE != NNODES )); then
  echo "this one-GPU-per-Spark bundle requires TP_SIZE=EP_SIZE=NNODES" >&2
  exit 2
fi

exec python3 -m sglang.launch_server \
  --model-path /model \
  --served-model-name glm-5.3-flash \
  --tp-size "$TP_SIZE" \
  --ep-size "$EP_SIZE" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_INIT_ADDR" \
  --context-length "$CONTEXT_LENGTH" \
  --quantization modelopt_fp4 \
  --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla \
  --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$CHUNKED_PREFILL_SIZE" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS" \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --media-url-max-file-size-mb 1024 \
  --enable-multimodal \
  --chat-template /opt/glm53/chat-template-mm.jinja \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port "$PORT"
