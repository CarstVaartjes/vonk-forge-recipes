#!/usr/bin/env bash
set -euo pipefail

# Single-Spark (TP=1) launcher for Inkling-2-Mini EXLR8 + DFlash.
# Works inside ghcr.io/kindlingai/gr8 (/opt/gr8) or a bare-metal gr8 checkout.

ENGINE_ROOT="${ENGINE_ROOT:-${GR8_ROOT:-/opt/gr8}}"
G6M4="${G6M4:-$ENGINE_ROOT/bin/g6m4_ink2}"
FRONT="${FRONT:-$ENGINE_ROOT/bin/g6front-rs}"
SHIM="${SHIM:-$ENGINE_ROOT/frontend/g6s2shim.py}"
PY="${PY:-python3}"

# gr8 strictly rejects unknown GR8_* environment variables; unset any wrapper vars.
unset GR8_ROOT GR8_IMAGE GR8_DIR GR8_REPO GR8_REF

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to your local kindlingai/inkling-2-mini-exlr8-k2-k3-k4 directory}"
SERVED_MODEL="${SERVED_MODEL:-inkling-2-mini}"
VARIANT="${VARIANT:-bit325}"
MTP="${MTP:-7}"
CAP="${CAP:-147456}"
SESSIONS="${SESSIONS:-8}"
CHUNK="${CHUNK:-8}"
PREFILL="${PREFILL:-4096}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
REUSE_PREFIX="${REUSE_PREFIX:-0}"
WORK_DIR="${WORK_DIR:-/tmp/ink2-gr8}"

export GR8_INK2_DF2_Q8="${GR8_INK2_DF2_Q8:-0}"
export GR8_INK2_PF_MX8_MOE="${GR8_INK2_PF_MX8_MOE:-2}"
export GR8_INK2_MX8_DENSE="${GR8_INK2_MX8_DENSE:-2}"
export GR8_INK2_MX8_LM_HEAD="${GR8_INK2_MX8_LM_HEAD:-0}"
export GR8_INK2_CUBLAS_PREFILL="${GR8_INK2_CUBLAS_PREFILL:-1}"
export GR8_INK2_TC_PREFILL_ATTN="${GR8_INK2_TC_PREFILL_ATTN:-1}"
export GR8_INK2_GPU_HEAD_PACK="${GR8_INK2_GPU_HEAD_PACK:-1}"

BIN_DIR="$(dirname "$G6M4")"
export LD_LIBRARY_PATH="${BIN_DIR}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
SOCK="$WORK_DIR/engine.sock"

cleanup() {
  trap - EXIT INT TERM
  kill "${FRONT_PID:-}" "${SHIM_PID:-}" "${ENG_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Inkling-2-Mini (variant=$VARIANT, mtp=$MTP, cap=$CAP, sessions=$SESSIONS, prefill=$PREFILL, port=$PORT)..."

ENG_ARGS=(
  --rank 0 --ips 127.0.0.1
  --v2-dense "$MODEL_DIR"
  --variant "$VARIANT"
  --serve "$WORK_DIR"
  --s2-epoch 1
  --s2-sessions "$SESSIONS"
  --cap "$CAP"
  --chunk "$CHUNK"
  --prefill "$PREFILL"
  --mtp "$MTP"
)
[ "$REUSE_PREFIX" = "1" ] && ENG_ARGS+=(--reuse-prefix)

# Zero extra DFlash flag needed: g6m4_ink2 auto-discovers the bundled DFlash drafter in MODEL_DIR.
"$G6M4" "${ENG_ARGS[@]}" &
ENG_PID=$!

for _ in $(seq 1 600); do
  [ -f "$WORK_DIR/ready.s2" ] && break
  kill -0 "$ENG_PID" 2>/dev/null || { echo "Engine exited early" >&2; exit 1; }
  sleep 0.2
done
[ -f "$WORK_DIR/ready.s2" ] || { echo "Timed out waiting for engine ready.s2" >&2; exit 1; }

"$PY" "$SHIM" \
  --dir "$WORK_DIR" \
  --sock "$SOCK" \
  --sessions "$SESSIONS" \
  --capacity-slots "$SESSIONS" \
  --max-model-len "$CAP" \
  --max-plp-prompt 2048 \
  --model-name "$SERVED_MODEL" \
  --model-dir "$MODEL_DIR" \
  --vocab-size 201024 \
  --eos-ids 200006,199999 &
SHIM_PID=$!

for _ in $(seq 1 100); do
  [ -S "$SOCK" ] && break
  kill -0 "$SHIM_PID" 2>/dev/null || { echo "Shim exited early" >&2; exit 1; }
  sleep 0.1
done
[ -S "$SOCK" ] || { echo "Timed out waiting for $SOCK" >&2; exit 1; }

"$FRONT" \
  --model-dir "$MODEL_DIR" \
  --engine-sock "$SOCK" \
  --host "$HOST" \
  --port "$PORT" \
  --dialect inkling \
  --served-model-name "$SERVED_MODEL" \
  --max-model-len "$CAP" \
  --enable-prompt-tokens-details \
  --default-reasoning-effort "$REASONING_EFFORT" \
  --allow-thinking-off \
  --reasoning-fields both &
FRONT_PID=$!

wait -n "$ENG_PID" "$SHIM_PID" "$FRONT_PID"
