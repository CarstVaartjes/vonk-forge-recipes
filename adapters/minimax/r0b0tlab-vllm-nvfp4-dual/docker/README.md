# Docker image

Runtime image:

```text
ghcr.io/r0b0tlab/vllm-minimax-m27-nvfp4-blackwell:cu130-sm120-arm64-v0.21.1
ghcr.io/r0b0tlab/vllm-minimax-m27-nvfp4-blackwell:latest-sm120-arm64
ghcr.io/r0b0tlab/vllm-minimax-m27-nvfp4-blackwell@sha256:9ee93af94a508bf7ef6125f6bf655c27b9082a892296d9db20ecdee6e360cb03
```

The image does not include model weights. Mount your licensed local copy of `nvidia/MiniMax-M2.7-NVFP4` at runtime.

## Compatible devices

Validated: dual NVIDIA GB10 / SM121.
Expected: Blackwell SM120/SM121 systems with sufficient aggregate memory for the ~130 GiB NVFP4 checkpoint plus KV cache.
Not an optimized claim for Ampere/Hopper or Marlin/emulation fallback.

## Verify FP4 support inside the image

```bash
docker run --rm --gpus all --ipc=host --network=host   --entrypoint python3   ghcr.io/r0b0tlab/vllm-minimax-m27-nvfp4-blackwell:cu130-sm120-arm64-v0.21.1 - <<'PY'
import torch, vllm, vllm._C_stable_libtorch
print('vllm', vllm.__version__)
print('cuda_available', torch.cuda.is_available())
print('capability', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
print('fp4_sm120', torch.ops._C.cutlass_scaled_mm_supports_fp4(120))
print('fp4_sm121', torch.ops._C.cutlass_scaled_mm_supports_fp4(121))
assert torch.cuda.is_available()
assert torch.ops._C.cutlass_scaled_mm_supports_fp4(120)
assert torch.ops._C.cutlass_scaled_mm_supports_fp4(121)
PY
```

## Serve smoke test

Replace `/path/to/MiniMax-M2.7-NVFP4` with your licensed local checkpoint directory.

```bash
docker run --rm --gpus all --ipc=host --network=host   -v /path/to/MiniMax-M2.7-NVFP4:/models/minimax-m27-nvfp4:ro   -e MODEL_PATH=/models/minimax-m27-nvfp4   -e PORT=30000   -e GPU_MEMORY_UTILIZATION=0.55   -e MAX_MODEL_LEN=4096   -e MAX_NUM_BATCHED_TOKENS=4096   -e MAX_NUM_SEQS=1   ghcr.io/r0b0tlab/vllm-minimax-m27-nvfp4-blackwell:cu130-sm120-arm64-v0.21.1
```
