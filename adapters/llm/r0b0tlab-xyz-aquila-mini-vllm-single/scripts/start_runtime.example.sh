#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/r0b0tdgx/projects/xyz-aquila-mini-nvfp4"
MODE="${1:?mode must be bf16 or nvfp4}"
MODEL_HOST="${2:?model path required}"
ATTEMPT="${3:?attempt id required}"
IMAGE="${IMAGE:-xyz-aquila-vllm-v025-sm121:702f4814}"
OUT="$ROOT/evidence/runs/$ATTEMPT"
mkdir -p "$OUT"
[[ -f "$MODEL_HOST/config.json" ]] || { echo "model config missing: $MODEL_HOST" >&2; exit 50; }
case "$MODE" in
  bf16)
    CONTAINER="xyz-aquila-bf16"
    PORT="18081"
    SERVED="xyz-aquila-bf16"
    QUANT=""
    MOE="triton"
    LANGUAGE_MODEL_ONLY="1"
    ;;
  nvfp4)
    CONTAINER="xyz-aquila-nvfp4"
    PORT="18082"
    SERVED="xyz-aquila-nvfp4"
    QUANT="modelopt_mixed"
    MOE="flashinfer_b12x"
    LANGUAGE_MODEL_ONLY="1"
    ;;
  *) echo "invalid mode: $MODE" >&2; exit 51 ;;
esac
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container already exists: $CONTAINER" >&2
  exit 52
fi
CACHE="$ROOT/artifacts/runtime-cache"
mkdir -p "$CACHE"/{huggingface,flashinfer,vllm}
{
  echo "mode=$MODE"
  echo "model=$MODEL_HOST"
  echo "image=$IMAGE"
  docker image inspect "$IMAGE" --format 'image_id={{.Id}}'
  sha256sum "$MODEL_HOST/config.json"
} > "$OUT/preflight.txt"
container_id=$(docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  -e "PORT=$PORT" \
  -e MODEL_ID=/models/model \
  -e "SERVED_MODEL_NAME=$SERVED" \
  -e KV_CACHE_DTYPE=fp8 \
  -e ATTENTION_BACKEND=flashinfer \
  -e "MOE_BACKEND=$MOE" \
  -e LINEAR_BACKEND=flashinfer_cutlass \
  -e "QUANTIZATION=$QUANT" \
  -e SPECULATIVE_CONFIG= \
  -e "MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}" \
  -e "MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}" \
  -e "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}" \
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}" \
  -e "LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY" \
  -e ENABLE_AUTO_TOOL_CHOICE=1 \
  -e TOOL_CALL_PARSER=qwen3_xml \
  -e REASONING_PARSER=qwen3 \
  -e MAX_JOBS=6 \
  -e FLASHINFER_NVCC_THREADS=2 \
  -v "$MODEL_HOST:/models/model:ro" \
  -v "$CACHE/huggingface:/root/.cache/huggingface" \
  -v "$CACHE/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE/vllm:/root/.cache/vllm" \
  "$IMAGE")
printf '%s\n' "$container_id" > "$OUT/container-id"
printf '%s\n' "$CONTAINER" > "$OUT/container-name"
printf '%s\n' "$PORT" > "$OUT/port"
printf '%s\n' "$SERVED" > "$OUT/served-model"

ready=0
for _ in $(seq 1 1200); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  state=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || true)
  if [[ "$state" == exited* || "$state" == dead* ]]; then break; fi
  sleep 1
done
docker logs "$CONTAINER" > "$OUT/server.log" 2>&1 || true
printf '%s\n' "$ready" > "$OUT/ready"
if [[ "$ready" -ne 1 ]]; then touch "$OUT/FAIL"; exit 53; fi
curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"
touch "$OUT/PASS"
printf 'container=%s port=%s served=%s\n' "$CONTAINER" "$PORT" "$SERVED"
