"""Fail the adapter build unless the image carries our packaging: NCCL, wrapper, kernel overlay."""

from __future__ import annotations

import os
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
OVERLAY = (
    VLLM / "v1/attention/backends/mla/sparse_mla_kernels.py",
    VLLM / "v1/attention/backends/mla/flashmla_sparse.py",
    VLLM / "v1/attention/ops/deepseek_v4_ops/sm12x_mqa.py",
    VLLM / "model_executor/layers/sparse_attn_indexer.py",
    VLLM / "model_executor/models/deepseek_v2.py",
)
REQUIRED_FILES = (
    Path("/opt/vonk/lib/libnccl.so.2"),
    Path("/usr/local/bin/vllm"),
    *OVERLAY,
)


def main() -> None:
    for path in REQUIRED_FILES:
        if not path.is_file():
            raise SystemExit(f"required GLM runtime file is missing: {path}")
    if Path("/opt/vonk/lib/libnccl.so.2").read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL runtime is not an ELF library")
    if not os.access("/usr/local/bin/vllm", os.X_OK):
        raise SystemExit("vLLM executable is not executable")


if __name__ == "__main__":
    main()
