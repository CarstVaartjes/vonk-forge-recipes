"""Fail the build unless the Qwen SM121 and NCCL runtime are complete."""

from __future__ import annotations

import os
from pathlib import Path


QSA = Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
    "qwen_sparse_attn_backend.py"
)
FALLBACK = QSA.with_name("qsa_fa_fallback.py")
NCCL = Path("/opt/vonk/lib/libnccl.so.2")


def main() -> None:
    if not QSA.is_file() or "qsa_fa_fallback" not in QSA.read_text():
        raise SystemExit("Qwen sparse-attention SM121 patch is missing")
    if not FALLBACK.is_file() or "triton_varlen_attn_func" not in FALLBACK.read_text():
        raise SystemExit("Qwen sparse-attention fallback kernel is missing")
    if not NCCL.is_file() or NCCL.read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL 2.30.7 runtime is missing or invalid")
    wrapper = Path("/opt/vonk/bin/sglang-serve")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("SGLang wrapper is missing or not executable")


if __name__ == "__main__":
    main()
