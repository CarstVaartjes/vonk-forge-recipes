"""Verify the MiMo-V2.5 Omni NVFP4-KV adapter packaging at build time.

Checks the end state upstream's start.sh asserts on both nodes (the nvfp4
DiffKV backend and non-causal support), by file content, since the build host
has no GPU.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
BACKEND = SITE / "v1/attention/backends/triton_attn_diffkv.py"
KERNEL = SITE / "v1/attention/ops/triton_unified_attention_diffkv.py"
WMMA = SITE / "v1/attention/ops/wmma_decode.py"
OMNI = SITE / "model_executor/models/mimo_v2_omni.py"
MTP = SITE / "model_executor/models/mimo_v2_mtp.py"
CONFIG_ROOT = Path("/opt/vonk/vllm-config/ray_non_carry_over_env_vars.json")
WRAPPER = Path("/opt/vonk/bin/vllm")
LAUNCHERS = (Path("/usr/local/bin/vllm"), Path("/usr/local/bin/ray"))


def main() -> None:
    for path in (BACKEND, KERNEL, WMMA, OMNI, MTP, CONFIG_ROOT, WRAPPER, *LAUNCHERS):
        if not path.is_file():
            raise SystemExit(f"required MiMo runtime file is missing: {path}")
    for path in (WRAPPER, *LAUNCHERS):
        if not os.access(path, os.X_OK):
            raise SystemExit(f"not executable: {path}")

    backend = BACKEND.read_text(encoding="utf-8")
    if not re.search(r"supported_kv_cache_dtypes[^=]*=\s*\[[^\]]*\"nvfp4\"", backend):
        raise SystemExit("the DiffKV backend does not list the nvfp4 KV cache dtype")
    if "def supports_non_causal" not in backend:
        raise SystemExit("the DiffKV backend lacks supports_non_causal")
    if "USE_CAUSAL" not in KERNEL.read_text(encoding="utf-8"):
        raise SystemExit("the DiffKV kernel lacks the USE_CAUSAL patch")
    if "NCCL_IB_HCA" not in CONFIG_ROOT.read_text(encoding="utf-8"):
        raise SystemExit("NCCL_IB_HCA is not excluded from Ray env carry-over")


if __name__ == "__main__":
    main()
