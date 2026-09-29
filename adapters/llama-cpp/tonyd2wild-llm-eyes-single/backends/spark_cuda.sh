#!/usr/bin/env bash
# NVIDIA DGX Spark (GB10, sm121, aarch64) vision server via llama.cpp CUDA.
# Same GGUF + mmproj path as the PC backend — the Spark just runs it on the GB10.
#
# Needs a CUDA build of llama-server. On the Spark:
#   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
#   cmake -B build -DGGML_CUDA=ON && cmake --build build -j
#   export PATH="$PWD/build/bin:$PATH"
set -euo pipefail
PORT="${PORT:-8081}"
REPO="${REPO:-unsloth/Qwen3.5-0.8B-GGUF}"
QUANT="${QUANT:-UD-Q8_K_XL}"        # GB10 has headroom; run a bigger quant for quality
MMPROJ="${MMPROJ:-mmproj-F16.gguf}"
NGL="${NGL:-99}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "[spark] ERROR: llama-server (CUDA build) not found on PATH."
  echo "[spark]   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"
  echo "[spark]   cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
  echo "[spark]   export PATH=\"\$PWD/build/bin:\$PATH\""
  exit 1
fi

echo "[spark] GB10 vision server on :$PORT — model ${REPO}:${QUANT} + ${MMPROJ}"
echo "[spark] endpoint: http://localhost:$PORT/v1"
exec llama-server \
  -hf "${REPO}:${QUANT}" \
  --mmproj "${REPO}/${MMPROJ}" \
  -ngl "$NGL" \
  --host 0.0.0.0 --port "$PORT" \
  --temp 0.6 --top-p 0.95 --top-k 20
