#!/bin/sh
# Controller launcher for the one-Spark MiniMax-M3.1-Flash EXL3 lane: the engine arguments are the recipe's.
set -eu
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec vllm "$@"
