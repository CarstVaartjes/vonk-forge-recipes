#!/usr/bin/env bash
# Final launch only. Explicit image ID and fresh run-dir required.
set -euo pipefail
exec bash "$(dirname "$0")/final_boot.sh" "$@"
