#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
PATCH_FILE="$SCRIPT_DIR/pr41797-vllm.diff"
EXPECTED_SHA256="01ace84baef35fea8d337f5b44623286fd150367140ae383cef846cfdd028a83"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[pr41797] ERROR: vendored immutable patch is missing: $PATCH_FILE" >&2
    exit 1
fi
actual_sha256="$(python3 - "$PATCH_FILE" <<'PY_HASH'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY_HASH
)"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "[pr41797] ERROR: vendored patch SHA-256 mismatch (got $actual_sha256)." >&2
    exit 1
fi
if [[ ! -d "$SITE_PACKAGES/vllm" ]]; then
    echo "[pr41797] ERROR: expected vLLM package tree is missing: $SITE_PACKAGES/vllm" >&2
    exit 1
fi

# A backend module alone does not prove the complete patch is present: the
# patch also changes MiMo backend selection, backend registration and the
# flash/unified attention paths. Skip only when every reverse hunk matches.
# Otherwise require every forward hunk before changing anything. A partial
# installation therefore fails closed instead of being mistaken for success.
cd "$SITE_PACKAGES"
if git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "[pr41797] Complete pinned PR #41797 patch is already applied."
    exit 0
fi

# Apply only after every patch anchor has passed. Do not fuzzy-apply a patch
# against a different or partially patched vLLM source tree.
if ! git apply --check "$PATCH_FILE"; then
    echo "[pr41797] ERROR: vLLM source is neither the exact patch base nor complete PR #41797 postimage." >&2
    echo "[pr41797] Rebuild with a reviewed base; partial or incompatible patch state is unsafe." >&2
    exit 1
fi
git apply "$PATCH_FILE"

if ! git apply --reverse --check "$PATCH_FILE"; then
    echo "[pr41797] ERROR: applied files do not match the complete PR #41797 postimage." >&2
    exit 1
fi
echo "[pr41797] Applied pinned vLLM PR #41797 patch from ab10addb603ac17611d55aa88b64acbf1da71373."
