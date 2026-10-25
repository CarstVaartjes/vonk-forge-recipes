"""Build check: import vLLM's MiniMax M2 model module as on a GB10 Spark.

The API server inspects MiniMaxM2ForCausalLM by importing
vllm.model_executor.models.minimax_m2. On a CUDA platform that import takes
CUDA-only branches (v1/attention/backends/fa_utils.py:17-23 imports
vllm.vllm_flash_attn), which a GPU-less builder never resolves. Resolve vLLM's
own non-NVML CUDA platform with the GB10 compute capability (12.1) instead, so
a missing CUDA extension on that path fails the image build, not the Spark.
The builder has no driver either, so the Dockerfile runs this check (only) with
the CUDA toolkit's libcuda stub as libcuda.so.1, which vllm._C links against.
"""

from __future__ import annotations

import importlib

import torch
import vllm.platforms as platforms


def main() -> None:
    if platforms._current_platform is not None:
        raise SystemExit("vLLM resolved its platform before the check could pin CUDA")
    torch.cuda.get_device_capability = lambda device=None: (12, 1)
    platforms.builtin_platform_plugins = {
        "cuda": lambda: "vllm.platforms.cuda.NonNvmlCudaPlatform"
    }
    if not platforms.current_platform.is_cuda():
        raise SystemExit(f"expected a CUDA platform, got {platforms.current_platform}")
    importlib.import_module("vllm.model_executor.models.minimax_m2")
    print("imported vllm.model_executor.models.minimax_m2 on a CUDA platform")


if __name__ == "__main__":
    main()
