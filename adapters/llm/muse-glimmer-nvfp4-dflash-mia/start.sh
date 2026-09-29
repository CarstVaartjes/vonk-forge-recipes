#!/usr/bin/env bash
#
# start.sh — serve Preyazz/Muse-Glimmer-30B-NVFP4 with vLLM on port 8888.
#
# Uses the vLLM fork xianbaoqian/vllm@tiezhen/new-model-support (PR
# vllm-project/vllm#51655), which adds native Muse-Glimmer support,
# the muse_glimmer reasoning/tool-call parsers and DFlash decoding.
#
# This script:
#   * creates a local venv and installs the vLLM fork (precompiled kernels) once
#   * downloads the model weights + DFlash draft head into the HF cache if missing
#   * starts the server with DFlash speculative decoding
#   * streams the vLLM log until the server answers, then returns to the shell
#
# The reasoning parser is required: Muse uses channel-scoped output framing
# (not <thinking> tags) and forces skip_special_tokens=False — without it the
# channels collapse and output is empty.
#
# 256k context: the model's native cap is 131072 (max_position_embeddings).
# Overshooting it needs BOTH of these:
#   * VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  (else vLLM refuses max_model_len > 131072)
#   * RoPE caches sized to max_model_len (patched in the fork:
#     muse_glimmer.py + qwen3_dflash.py). Without the patch, positions > 131071
#     index past the precomputed cos/sin cache and crash. Output between
#     131072..262144 is RoPE extrapolation (untrained) — use with caution.
#
set -Eeuo pipefail

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
SERVED_MODEL_NAME="muse-glimmer-30b"
MAX_MODEL_LEN=262144   # 256k context (RoPE caches patched to size to this; see below)
GPU_MEM_UTIL=0.42		# fp8 KV measured 9.21x256k (2,414,048 slots) = 15% margin over 8
MODEL_REPO="RedHatAI/Muse-Glimmer-30B-NVFP4"             # W4A4 variant (same llm-compressor recipe family; vision intact)
DRAFT_REPO="meta-models/Muse-Glimmer-30B-assistant"   # DFlash draft head (~5.1 GB)
VLLM_REPO_URL="https://github.com/xianbaoqian/vllm"
VLLM_BRANCH="tiezhen/new-model-support"

VENV="$SCRIPT_DIR/.venv"
VLLM_DIR="$SCRIPT_DIR/vllm"
LOG="$SCRIPT_DIR/vllm.log"
PID_FILE="$SCRIPT_DIR/vllm.pid"
INSTALL_MARKER="$VLLM_DIR/.installed-commit"

log() { printf '\033[1;36m[start.sh]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[start.sh] ERROR:\033[0m %b\n' "$*" >&2; exit 1; }

# --- 0. already running? ---------------------------------------------------
if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  log "a server is already responding on port $PORT — nothing to do."
  exit 0
fi
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  log "server is already starting (pid $(cat "$PID_FILE")) — log: $LOG"
  exit 0
fi

command -v curl   >/dev/null || die "curl is required."
command -v python3 >/dev/null || die "python3 is required."

# --- 1. venv ----------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv at $VENV"
  python3 -m venv "$VENV"
fi

# --- 2. vLLM fork (built once per fetched commit) ---------------------------
if [ ! -d "$VLLM_DIR/.git" ]; then
  log "cloning vLLM fork ($VLLM_BRANCH)…"
  git clone --depth 1 -b "$VLLM_BRANCH" "$VLLM_REPO_URL" "$VLLM_DIR"
else
  git -C "$VLLM_DIR" fetch origin "$VLLM_BRANCH" --depth 1
  git -C "$VLLM_DIR" checkout -B "$VLLM_BRANCH" "origin/$VLLM_BRANCH"
fi
COMMIT="$(git -C "$VLLM_DIR" rev-parse HEAD)"
if [ -f "$INSTALL_MARKER" ] && [ "$(cat "$INSTALL_MARKER")" = "$COMMIT" ]; then
  log "vLLM fork already installed (commit ${COMMIT:0:12}) — skipping build"
else
  log "installing vLLM fork (commit ${COMMIT:0:12}) with precompiled kernels — this takes a while…"
  ( cd "$VLLM_DIR" && VLLM_USE_PRECOMPILED=1 "$VENV/bin/pip" install -e . )
  echo "$COMMIT" > "$INSTALL_MARKER"
fi
[ -x "$VENV/bin/vllm" ] || die "vllm binary not found in the venv after install"
log "ensuring ninja (required by vLLM's JIT)"
"$VENV/bin/pip" install -q ninja

# --- 3. weights into the HF cache -------------------------------------------
export REPO_ID="$MODEL_REPO"
log "checking/downloading $MODEL_REPO into the HF cache…"
MODEL_PATH="$("$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ["REPO_ID"]))
PY
)"
export REPO_ID="$DRAFT_REPO"
log "checking/downloading $DRAFT_REPO (DFlash draft head)…"
DRAFT_PATH="$("$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ["REPO_ID"]))
PY
)"

# --- 4. launch the server ----------------------------------------------------
# venv bin on PATH so vLLM's JIT finds `ninja`
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
: > "$LOG"

SPEC_CFG="{\"method\": \"dflash\", \"model\": \"$DRAFT_PATH\", \"num_speculative_tokens\": 16}"
SERVE_ARGS=(
  serve
  "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --reasoning-parser muse_glimmer
  --tool-call-parser muse_glimmer
  --enable-auto-tool-choice
  --speculative-config "$SPEC_CFG"
  --kv-cache-dtype fp8_e4m3   # doubles KV capacity; 8x256k = ~16.4 GiB of KV
  # FlashInfer autotune's dummy run (num_tokens=2048, randomized inputs) drives
  # the DFlash draft attention out of bounds -> CUDA device-side assert.
  # Autotune stays off even though fp8 KV puts decode attention on FlashInfer.
  --no-enable-flashinfer-autotune
)

log "starting vLLM on port $PORT …"
nohup "$VENV/bin/vllm" "${SERVE_ARGS[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
disown "$SERVER_PID" 2>/dev/null || true   # keep running after this script exits

# --- 5. stream the vLLM log until the server answers, then return -----------
tail -n +1 -F "$LOG" &
TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT

while :; do
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$TAIL_PID" 2>/dev/null || true
    echo "" >&2
    die "vLLM exited during startup — last log lines:\n$(tail -n 60 "$LOG")"
  fi
  sleep 2
done

kill "$TAIL_PID" 2>/dev/null || true
trap - EXIT

echo ""
log "Muse-Glimmer-30B-NVFP4 is serving"
log "  base URL : http://0.0.0.0:$PORT  (OpenAI-compatible API)"
log "  model    : $SERVED_MODEL_NAME"
log "  decoding : dflash ($DRAFT_REPO, 15 speculative tokens)"
log "  log file : $LOG"
log "  stop     : kill \$(cat $PID_FILE)"
echo ""
exit 0
