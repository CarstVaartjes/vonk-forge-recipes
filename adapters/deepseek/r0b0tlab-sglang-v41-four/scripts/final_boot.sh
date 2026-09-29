#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE=${1:?profile required}; shift
exec python3 scripts/final_runtime.py launch --profile "$PROFILE" "$@"
