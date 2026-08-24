"""Fail the adapter build unless the image contains the matched GLM stack."""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
REQUIRED_MARKERS = {
    SITE / "vllm/utils/deep_gemm.py": "glm52_sm12x_patch",
    SITE / "vllm/model_executor/layers/sparse_attn_indexer.py": "glm52_sm12x_patch",
    SITE / "vllm/v1/attention/backends/mla/indexer.py": (
        "MTP spec tokens can extend a request one block past max_model_len"
    ),
    SITE / "b12x/fused_indexer.py": "glm52_patch",
}
REQUIRED_FILES = (
    SITE / "vllm/v1/attention/backends/mla/sparse_mla_kernels.py",
    SITE / "vllm/v1/attention/backends/mla/sm12x_sparse_mla_attn.py",
    SITE / "vllm/v1/attention/ops/deepseek_v4_ops/sm12x_mqa.py",
    SITE / "vllm/v1/attention/ops/deepseek_v4_ops/b12x_sparse_helpers.py",
    Path("/opt/vonk/lib/libnccl.so.2"),
    Path("/usr/local/bin/vllm"),
)


def main() -> None:
    for path, marker in REQUIRED_MARKERS.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"required GLM runtime patch is missing: {path}")
    for path in REQUIRED_FILES:
        if not path.is_file():
            raise SystemExit(f"required GLM runtime file is missing: {path}")
    if Path("/opt/vonk/lib/libnccl.so.2").read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL runtime is not an ELF library")
    if not os.access("/usr/local/bin/vllm", os.X_OK):
        raise SystemExit("vLLM executable is not executable")


if __name__ == "__main__":
    main()
