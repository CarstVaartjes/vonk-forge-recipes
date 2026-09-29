#!/usr/bin/env bash
# Final optimized TP=2 MiniMax M2.7 NVFP4 launch script.
# Uses FLASHINFER_CUTLASS backend with torch.compile (Inductor), no CUDA graphs.
#
# Breakthrough: CUDA graph replay deadlocks SHM broadcast between cross-node
# TP workers. Compile-only mode avoids this while keeping Inductor optimizations.
# FLASHINFER_CUTLASS MoE gives ~3-5% additional over plain VLLM_CUTLASS.
#
# Results: tg128 = 25.13 t/s = 103.4% of public baseline (24.30)
#
# Usage: ./scripts/launch_gb10_dual_tp2_optimized.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/path/to/MiniMax-M2.7-NVFP4}
PY=${PYTHON_BIN:-python3}
PORT=${PORT:-30000}
GPU_UTIL=${GPU_UTIL:-0.85}
MAX_SEQS=${MAX_SEQS:-4}
BATCHED=${BATCHED:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-196608}

export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=1
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export VLLM_HOST_IP=${VLLM_HOST_IP:-<HEAD_NODE_QSFP_IP>}
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

echo "=== MiniMax M2.7 NVFP4 OPTIMIZED TP=2 ==="
echo "Backend: FLASHINFER_CUTLASS (MoE + linear)"
echo "Compile: torch.compile (Inductor) mode 3"
echo "CUDA graphs: disabled (avoids SHM broadcast deadlock)"
echo "Topology: TP=2 dual-GB10 Ray"
echo "GPU util: $GPU_UTIL | Max seqs: $MAX_SEQS"
echo ""

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --quantization modelopt_fp4 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-batched-tokens "$BATCHED" \
  --max-num-seqs "$MAX_SEQS" \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --moe-backend flashinfer_cutlass \
  --attention-backend flashinfer \
  --served-model-name minimax-m2.7-nvfp4 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --compilation-config '{"mode": 3, "cudagraph_mode": "none", "inductor_compile_config": {"combo_kernels": false, "benchmark_combo_kernel": false, "max_autotune": false, "max_autotune_gemm": false}}'
