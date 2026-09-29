#!/bin/bash
set -euo pipefail

PREFIX="[inkling-dspark-vllm]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
PATCHER="$MOD_DIR/patch_inkling_dspark.py"

TARGET=""
for candidate in \
    "$VLLM_ROOT/models/inkling/nvidia/model.py" \
    "$VLLM_ROOT/model_executor/models/inkling/nvidia/model.py"; do
    if [[ -f "$candidate" ]]; then
        TARGET="$candidate"
        break
    fi
done

if [[ -z "$TARGET" ]]; then
    echo "$PREFIX Inkling NVIDIA model module was not found under $VLLM_ROOT." >&2
    exit 1
fi

python3 "$PATCHER" --check "$TARGET"
python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

find "$(dirname "$TARGET")" -name __pycache__ -type d \
    -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX Inkling now exposes vLLM auxiliary hidden states for DSpark."
