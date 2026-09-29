#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="nvidia/Qwen3.6-27B-NVFP4"
SERVED_MODEL_NAME="qwen36-27b-nvidia-nvfp4-dflash"
IMAGE="vllm/vllm-openai:nightly-aarch64"
CONTAINER_NAME="qwen3.6-27b-nvidia-nvfp4-dflash"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

if [[ ! -f "${WORK_DIR}/chat_template.jinja" ]]; then
  echo "Missing chat_template.jinja in ${WORK_DIR}"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting vLLM container for ${MODEL_ID} (DFlash speculative decoding)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container (DFlash)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e CUTE_DSL_ARCH=sm_121a \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  -v "${WORK_DIR}/chat_template.jinja:/workspace/chat_template.jinja" \
  -v "${WORK_DIR}:/workspace" \
  --entrypoint vllm \
  "${IMAGE}" \
  serve \
  "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --moe-backend flashinfer_b12x \
  --attention-backend flash_attn \
  --kv-cache-dtype bfloat16 \
  --gpu-memory-utilization 0.84 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --generation-config vllm \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}' \
  --chat-template /workspace/chat_template.jinja \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "vLLM is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"

echo "vLLM is ready and responding; shell is now free."
