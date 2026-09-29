#!/bin/bash
set -euo pipefail

PREFIX="[inkling-agent-parser-fix]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
TARGET="$PYTHON_ROOT/vllm/parser/inkling.py"

if [[ ! -f "$TARGET" ]]; then
  echo "$PREFIX Inkling parser not found: $TARGET" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_inkling_parser.py" --check "$TARGET"
python3 "$MOD_DIR/patch_inkling_parser.py" "$TARGET"
python3 "$MOD_DIR/patch_inkling_parser.py" --check "$TARGET"
find "$(dirname "$TARGET")" -name __pycache__ -type d \
  -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX terminal control-token filtering is active."
