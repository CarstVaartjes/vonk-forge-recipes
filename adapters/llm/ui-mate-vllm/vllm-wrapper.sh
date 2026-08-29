#!/bin/sh
set -eu

mkdir -p \
  "$HOME" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$HF_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$UV_CACHE_DIR"

exec vllm "$@"
