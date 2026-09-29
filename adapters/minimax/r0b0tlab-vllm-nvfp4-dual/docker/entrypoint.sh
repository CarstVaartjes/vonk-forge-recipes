#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${MODEL_PATH:-/models/minimax-m27-nvfp4}
PORT=${PORT:-8000}
python3 - <<'PY'
import torch, vllm, vllm._C_stable_libtorch
print('vllm', vllm.__version__, vllm.__file__)
print('cuda', torch.cuda.is_available())
print('fp4_121', torch.ops._C.cutlass_scaled_mm_supports_fp4(121))
assert torch.cuda.is_available()
assert torch.ops._C.cutlass_scaled_mm_supports_fp4(121)
PY
exec python3 -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --served-model-name minimax-m2.7-nvfp4 \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code --quantization modelopt_fp4 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --moe-backend "${MOE_BACKEND:-flashinfer_cutlass}" \
  --enable-prefix-caching \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --disable-custom-all-reduce \
  --compilation-config '{"mode":3,"cudagraph_mode":"none","inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"max_autotune":false,"max_autotune_gemm":false}}'
