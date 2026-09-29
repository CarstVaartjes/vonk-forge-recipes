#!/usr/bin/env bash
# =============================================================================
#  start.sh — NVIDIA/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 on DGX Spark
#             (GB10 / SM121) with DSpark speculative decoding
#
#  Recipe: docs.sglang.io cookbook "Nemotron3.5-Lightning"
#          hw=dgx-spark, variant=default, quant=nvfp4,
#          strategy=dspark, nodes=single
#
#  Default: public prebuilt Docker image — no source build required.
#  Spark: tp-size 1, mem-fraction-static 0.78, cuda-graph-max-bs-decode 4.
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
DRAFT_MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
# Official day-0 image from the SGLang cookbook for Nemotron 3.5 Lightning.
IMAGE="${IMAGE:-lmsysorg/sglang:dev-nemotron3-5-lightning}"
CONTAINER_NAME="nemotron-3.5-lightning-dspark"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
WORK_DIR="$(pwd)"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
PID_FILE="${WORK_DIR}/.sglang.pid"
LOG_FILE="${WORK_DIR}/.sglang.log"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# SGLang tunables (cookbook DGX-Spark DSPARK operating point as defaults)
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.78}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-4}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-3}"

# ---- Argument parsing -------------------------------------------------------
DOWNLOAD_ONLY=false
case "${1:-}" in
  --download-only)
    DOWNLOAD_ONLY=true
    shift
    ;;
  -h|--help)
    echo "Usage: $0 [--download-only]"
    echo ""
    echo "  --download-only    Download the model (+ DSpark draft) to"
    echo "                     ~/.cache/huggingface/hub then exit without"
    echo "                     starting SGLang."
    echo ""
    echo "  Environment variables:"
    echo "    PORT                   Server port (default: 8888)"
    echo "    HOST                   Bind address for SGLang (default: 0.0.0.0)"
    echo "    MEM_FRACTION_STATIC    SGLang static mem fraction (default: 0.78)"
    echo "    CUDA_GRAPH_MAX_BS_DECODE  (default: 4)"
    echo "    DSPARK_BLOCK_SIZE      spec draft block size (default: 3)"
    echo "    HF_TOKEN               Hugging Face token for gated models"
    echo "    IMAGE                  Docker image (default: lmsysorg/sglang:dev-nemotron3-5-lightning)"
    echo "    HF_HOME                HF cache dir (default: ~/.cache/huggingface)"
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1"
    echo "Usage: $0 [--download-only]"
    exit 1
    ;;
esac

# ---- Prerequisite checks ----------------------------------------------------
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker is required"; exit 1; }
command -v curl   >/dev/null 2>&1 || { echo "FATAL: curl is required";   exit 1; }

# Docker daemon
if ! docker info >/dev/null 2>&1; then
  echo "FATAL: Docker daemon is not reachable (docker info failed)."
  exit 1
fi

# NVIDIA driver on the host
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "FATAL: No NVIDIA driver found (nvidia-smi -L failed)."
  echo "       Install the NVIDIA driver for this GPU before continuing."
  exit 1
fi

# NVIDIA container runtime (required for 'docker run --gpus all')
if ! command -v nvidia-container-runtime >/dev/null 2>&1 \
  && ! command -v nvidia-container-runtime-hook >/dev/null 2>&1 \
  && ! docker info 2>/dev/null | grep -qi 'nvidia'; then
  echo "FATAL: NVIDIA container runtime not found — '--gpus all' will fail."
  echo "       Install nvidia-container-toolkit, e.g.:"
  echo "         sudo apt-get install -y nvidia-container-toolkit"
  echo "         sudo systemctl restart docker"
  exit 1
fi

# Host memory (unified memory: model + compile + KV all share host RAM)
MEM_TOTAL_GIB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
if (( MEM_TOTAL_GIB < 80 )); then
  echo "FATAL: host has only ${MEM_TOTAL_GIB} GiB RAM; needs ~119 GiB (DGX Spark class)."
  exit 1
fi
if (( MEM_TOTAL_GIB < 110 )); then
  echo "WARN:  ${MEM_TOTAL_GIB} GiB RAM is below the DGX Spark class (~119 GiB)."
  echo "       Consider a lower MEM_FRACTION_STATIC."
fi

# Free disk for image (~20 GB) + two weight sets (NVFP4 + DSpark draft)
FREE_GIB=$(( $(df -Pk "${WORK_DIR}" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
if (( FREE_GIB < 30 )); then
  echo "FATAL: only ${FREE_GIB} GiB free on ${WORK_DIR} — need >= 30 GiB."
  exit 1
fi

# Ports (host network: must be free on the host)
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
  echo "FATAL: port ${PORT} is already in use — a server may already be running."
  echo "       Stop it first, or pick another port via PORT=<port>."
  exit 1
fi

# ---- Env exports (also passed to container) ---------------------------------
export HF_HOME
export HF_TOKEN
mkdir -p "${HF_HOME}"

is_hf_model_id() { [[ "${1}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; }

hf_cache_repo_dir() { echo "${HF_HOME}/hub/models--${1//\//--}"; }

# ---- Model caching helpers --------------------------------------------------
model_is_fully_cached() {
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${1}")"
  [[ -d "${cache_dir}/snapshots" ]] || return 1
  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    [[ -f "${snapshot}/config.json" ]] || continue
    if [[ -f "${snapshot}/model.safetensors" ]] \
      || [[ -f "${snapshot}/model.safetensors.index.json" ]] \
      || compgen -G "${snapshot}/model-"*.safetensors >/dev/null \
      || [[ -f "${snapshot}/consolidated.safetensors" ]]; then
      return 0
    fi
  done
  return 1
}

download_model() {
  local model_id="$1"
  echo ""
  echo "  >> Downloading ${model_id} …"
  echo "     (cache: ${HF_HOME})"
  echo "     This can take a while for large models."

  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" hf download "${model_id}" \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" huggingface-cli download "${model_id}" \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  # Fallback: download inside Docker
  docker run --rm \
    --entrypoint python3 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN') or None)
"
}

ensure_model() {
  local model_id="$1" label="$2"
  if model_is_fully_cached "${model_id}"; then
    echo "  [✓] ${label} (${model_id}) is cached"
  else
    echo "  [↓] ${label} not cached — downloading …"
    download_model "${model_id}"
    if model_is_fully_cached "${model_id}"; then
      echo "  [✓] ${label} download complete"
    else
      echo "  [✗] ${label} download appears incomplete — check logs above"
      exit 1
    fi
  fi
}

# ---- Download model (idempotent) --------------------------------------------
echo "=============================================================================="
echo "  NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4  —  DSpark on DGX Spark"
echo "  $(date)"
echo "=============================================================================="
echo ""
echo "Checking model cache …"

ensure_model "${MODEL_ID}" "Nemotron-3.5-Lightning NVFP4"
ensure_model "${DRAFT_MODEL_ID}" "DSpark draft model"
echo ""

# ---- Early exit for download-only mode --------------------------------------
if ${DOWNLOAD_ONLY}; then
  echo "=============================================================================="
  echo "  Models are cached. Exiting (--download-only)."
  echo "=============================================================================="
  exit 0
fi

# ---- Container lifecycle ----------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing existing container ${CONTAINER_NAME} …"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo ""
echo "Pulling ${IMAGE} ..."
if ! docker pull "${IMAGE}"; then
  echo "FATAL: failed to pull ${IMAGE}"
  exit 1
fi
echo ""

echo "Starting SGLang server for ${MODEL_ID} (DSpark)"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo ""

# Resolve snapshot paths (must be done before docker run)
resolve_snapshot() {
  local model_id="$1"
  local cache_dir snap
  cache_dir="$(hf_cache_repo_dir "${model_id}")"
  snap="$(ls "${cache_dir}/snapshots/" 2>/dev/null | head -1)"
  if [[ -z "${snap}" ]]; then
    echo "FATAL: Could not find snapshot for ${model_id} in ${cache_dir}/snapshots/"
    echo "       Run with --download-only first, or check your HF_HOME."
    exit 1
  fi
  echo "${cache_dir}/snapshots/${snap}"
}
MODEL_SNAPSHOT="$(resolve_snapshot "${MODEL_ID}")"
DRAFT_SNAPSHOT="$(resolve_snapshot "${DRAFT_MODEL_ID}")"
# Paths as seen inside the container (HF_HOME is bind-mounted at /root/.cache/huggingface)
MODEL_IN_CONTAINER="/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/${MODEL_SNAPSHOT##*/}"
DRAFT_IN_CONTAINER="/root/.cache/huggingface/hub/models--${DRAFT_MODEL_ID//\//--}/snapshots/${DRAFT_SNAPSHOT##*/}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (${MODEL_ID} + DSpark)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  --ipc host \
  --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${WORK_DIR}:/workspace" \
  --workdir /workspace \
  "${IMAGE}" \
  sglang serve \
    --model-path "${MODEL_IN_CONTAINER}" \
    --mamba-ssm-dtype float16 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}" \
    --reasoning-parser nemotron_3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${DRAFT_IN_CONTAINER}" \
    --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id:0:12})"
echo "Log: ${LOG_FILE}"
echo ""

# ---- Wait for readiness -----------------------------------------------------
log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]]; then
    kill "${log_follow_pid}" 2>/dev/null || true
    wait "${log_follow_pid}" 2>/dev/null || true
    log_follow_pid=""
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for HTTP readiness at ${READY_URL}"
echo "--- container logs ---"

docker logs -f "${CONTAINER_NAME}" 2>&1 &
log_follow_pid=$!

while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo ""
    echo "SGLang container exited before becoming ready"
    exit 1
  fi
  sleep 2
done

cleanup

echo ""
echo "=============================================================================="
echo "  SGLang is ready!"
echo "  Model: ${MODEL_ID} + DSpark draft (${DRAFT_MODEL_ID})"
if [[ "${HOST}" == "0.0.0.0" || "${HOST}" == "::" ]]; then
  echo "  OpenAI-compatible endpoint:  http://127.0.0.1:${PORT}/v1  (bound on ${HOST})"
else
  echo "  OpenAI-compatible endpoint:  http://${HOST}:${PORT}/v1"
fi
echo ""
echo "  Recommended client params:"
echo "    temperature=0.6  top_p=0.95  top_k=20"
echo '    chat_template_kwargs: {"enable_thinking": true}'
echo "=============================================================================="
