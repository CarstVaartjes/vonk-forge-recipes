#!/bin/bash
set -euo pipefail

PREFIX="[inkling-warmup-cap]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
TARGET="$PYTHON_ROOT/vllm/v1/worker/gpu/warmup.py"

if [[ ! -f "$TARGET" ]]; then
  echo "$PREFIX vLLM warmup module not found: $TARGET" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_warmup.py" "$TARGET"
python3 "$MOD_DIR/patch_warmup.py" --check "$TARGET"
find "$(dirname "$TARGET")" -name __pycache__ -type d \
  -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX synthetic startup batch cap is active (runtime max_num_seqs is unchanged)."
