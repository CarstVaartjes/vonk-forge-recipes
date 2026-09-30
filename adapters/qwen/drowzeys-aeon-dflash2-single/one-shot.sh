#!/usr/bin/env bash
# One-shot: community compile of Qwen3.8-27B NVFP4 + DFlash2 n=7 + AEON uncensored
# + cyber unlock + 1M YaRN on ONE NVIDIA DGX Spark.
#
# This is a wiring script. We did not train a 27B. Read CREDITS.md.
#
#   I_AGREE=1 bash one-shot.sh
#   I_AGREE=1 bash one-shot.sh --speed     # 16k / seqs 64 instead of 1M
#
# GMU never above 0.85.
set -euo pipefail

KEYS_IMAGE="${KEYS_IMAGE:-ghcr.io/drowzeys/keys-qwen38-27b-nvfp4-dflash2-ablit-cyber-unlock-1m-yarn-single-dgxspark:latest}"
AEON_IMAGE="${AEON_IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"
BODY_REPO="${BODY_REPO:-sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4}"
DRAFT_REPO="${DRAFT_REPO:-incoai/Qwen3.8-27B-DFlash2}"
GIT_REPO="${GIT_REPO:-https://github.com/drowzeys/keys-Qwen3.8-27B-NVFP4-DFlash2-Ablit-Cybersecurity-Unlock-1M-Context-YARN-Single-DGXSpark}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4}"
DRAFT="${DRAFT:-$HOME/models/Qwen3.8-27B-DFlash2}"
PORT="${PORT:-8000}"
NAME="${NAME:-aeon-b5}"
SEAT="1m"
SKIP_PULL=0
SKIP_DL=0
NO_WAIT=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \?//'
  echo "Flags: --speed  --skip-pull  --skip-download  --no-wait  --i-agree  -h"
  echo "Env:   I_AGREE=1  MODEL  DRAFT  PORT  NAME  KEYS_IMAGE  AEON_IMAGE"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed) SEAT="16k"; shift ;;
    --skip-pull) SKIP_PULL=1; shift ;;
    --skip-download|--skip-dl) SKIP_DL=1; shift ;;
    --no-wait) NO_WAIT=1; shift ;;
    --i-agree) I_AGREE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${I_AGREE:-}" != "1" ]]; then
  echo "This stack is uncensored (AEON-7 ablit) plus a cybersecurity-unlock template." >&2
  echo "Read RESPONSIBLE_USE.md. Re-run with I_AGREE=1 or --i-agree." >&2
  exit 3
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$HERE/yarn_hf_overrides.json" ]]; then
  DEST="${CLONE_DIR:-$HOME/keys-qwen38-27b-1m-spark}"
  echo "[0/6] overlays missing — cloning recipe → $DEST"
  if [[ ! -f "$DEST/yarn_hf_overrides.json" ]]; then
    git clone --depth 1 "$GIT_REPO" "$DEST"
  fi
  HERE="$DEST"
fi

echo "== keys-Qwen3.8 27B NVFP4 DFlash2 Ablit+Cyber unlock 1M YaRN  (one DGX Spark) =="
echo "Community compile. Originals: Qwen · AEON-7 · sakamakismile · Inco/Z Lab · vLLM."
echo "Recipe: $HERE"
echo "Seat:   $SEAT   GMU cap 0.85"
echo

command -v docker >/dev/null || { echo "need docker" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "need nvidia-smi (DGX Spark / GPU host)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon not reachable" >&2; exit 1; }

echo "[1/6] GPU"
nvidia-smi -L | head -5

pull_image() {
  local img="$1"
  echo "  docker pull $img"
  docker pull "$img"
}

echo "[2/6] Prebuilt runtime (AEON-7 aeon-vllm-ultimate retag)"
IMAGE="$KEYS_IMAGE"
if [[ "$SKIP_PULL" -eq 1 ]]; then
  echo "  skip pull (IMAGE=$IMAGE)"
else
  if ! pull_image "$KEYS_IMAGE"; then
    echo "  Keys GHCR not pullable (often still private) — falling back to public AEON-7 image"
    pull_image "$AEON_IMAGE"
    IMAGE="$AEON_IMAGE"
  fi
fi
export IMAGE

hf_get() {
  local repo="$1" dest="$2"
  mkdir -p "$dest"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --local-dir "$dest"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" --local-dir "$dest"
  else
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$repo", local_dir="$dest")
PY
  fi
}

has_body() {
  [[ -f "$MODEL/model.safetensors" || -f "$MODEL/model.safetensors.index.json" ]] && [[ -f "$MODEL/config.json" ]]
}
has_draft() {
  [[ -f "$DRAFT/config.json" ]] && { [[ -f "$DRAFT/model.safetensors" || -f "$DRAFT/model.safetensors.index.json" ]]; }
}

echo "[3/6] Weights from original repos (not re-hosted here)"
if [[ "$SKIP_DL" -eq 1 ]]; then
  echo "  skip download"
else
  if has_body; then
    echo "  body already at $MODEL"
  else
    echo "  $BODY_REPO → $MODEL"
    hf_get "$BODY_REPO" "$MODEL"
  fi
  if has_draft; then
    echo "  drafter already at $DRAFT"
  else
    echo "  $DRAFT_REPO → $DRAFT"
    hf_get "$DRAFT_REPO" "$DRAFT"
  fi
fi
has_body || { echo "missing NVFP4 body at $MODEL" >&2; exit 1; }
has_draft || { echo "missing DFlash2 at $DRAFT" >&2; exit 1; }

echo "[4/6] Launch $SEAT"
export MODEL DRAFT NAME PORT
if [[ "$SEAT" == "16k" ]]; then
  bash "$HERE/serve_b5_16k.sh"
else
  bash "$HERE/serve_b5_on_4.sh"
fi

if [[ "$NO_WAIT" -eq 1 ]]; then
  echo "[5/6] skip wait (--no-wait). Health: curl -s http://127.0.0.1:${PORT}/v1/models"
  exit 0
fi

echo "[5/6] Wait for API (first boot ~8–12 min: weights + graphs)"
python3 - <<PY
import json, sys, time, urllib.request
url = "http://127.0.0.1:${PORT}/v1/models"
want = 16384 if "${SEAT}" == "16k" else 1048576
t0 = time.time()
while time.time() - t0 < 900:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.loads(r.read())
        ml = d["data"][0].get("max_model_len")
        print(f"  ready max_model_len={ml} after {time.time()-t0:.0f}s", flush=True)
        if int(ml) < want:
            print(f"  warning: expected {want}", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"  wait {time.time()-t0:.0f}s {type(e).__name__}", flush=True)
        time.sleep(10)
print("timeout waiting for /v1/models", file=sys.stderr)
sys.exit(1)
PY

echo "[6/6] Smoke"
python3 - <<PY
import json, urllib.request
body = {
    "model": "aeon",
    "messages": [{"role": "user", "content": "Reply with the single word PONG"}],
    "max_tokens": 8,
    "temperature": 0,
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    "http://127.0.0.1:${PORT}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as r:
    d = json.loads(r.read())
msg = d["choices"][0]["message"]
text = (msg.get("content") or "") + (msg.get("reasoning") or "")
print("  ", repr(text[:120]), d.get("usage"))
if "PONG" not in text.upper():
    raise SystemExit("smoke did not contain PONG")
print("SMOKE_OK")
PY

echo
echo "Endpoint  http://127.0.0.1:${PORT}/v1   model=aeon"
echo "Credits   $HERE/CREDITS.md"
echo "Done. Explore the original repos — this was only the wiring."
