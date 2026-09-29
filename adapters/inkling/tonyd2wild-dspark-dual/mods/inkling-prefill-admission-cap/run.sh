#!/bin/bash
set -euo pipefail

PREFIX="[inkling-prefill-admission-cap]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
TARGET="$PYTHON_ROOT/vllm/v1/core/sched/scheduler.py"

if [[ ! -f "$TARGET" ]]; then
  echo "$PREFIX vLLM scheduler not found: $TARGET" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_scheduler.py" "$TARGET"
python3 "$MOD_DIR/patch_scheduler.py" --check "$TARGET"
find "$(dirname "$TARGET")" -name __pycache__ -type d \
  -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX new-prefill cap is active; runtime max_num_seqs is unchanged."
