"""Fail the build unless the Qwen SM121 and NCCL runtime are complete."""

from __future__ import annotations

import os
from pathlib import Path


QSA = Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
    "qwen_sparse_attn_backend.py"
)
FALLBACK = QSA.with_name("qsa") / "sm121_varlen.py"
NCCL = Path("/opt/vonk/lib/libnccl.so.2")


def main() -> None:
    qsa_source = QSA.read_text() if QSA.is_file() else ""
    if (
        "qsa.sm121_varlen" not in qsa_source
        or "SM121 must not use TRT-LLM sparse decode" not in qsa_source
    ):
        raise SystemExit("Qwen sparse-attention SM121 patch is missing")
    if not FALLBACK.is_file() or "qsa_sm121_varlen_attention" not in FALLBACK.read_text():
        raise SystemExit("Qwen sparse-attention fallback kernel is missing")
    if not NCCL.is_file() or NCCL.read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL 2.30.7 runtime is missing or invalid")
    wrapper = Path("/opt/vonk/bin/sglang-serve")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("SGLang wrapper is missing or not executable")


if __name__ == "__main__":
    main()
