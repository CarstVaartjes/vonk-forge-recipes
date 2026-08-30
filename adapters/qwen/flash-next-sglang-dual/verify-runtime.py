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
SRT = QSA.parents[2]


def main() -> None:
    qsa_source = QSA.read_text() if QSA.is_file() else ""
    if (
        "qsa.sm121_varlen" not in qsa_source
        or "SM121 must not use TRT-LLM sparse decode" not in qsa_source
    ):
        raise SystemExit("Qwen sparse-attention SM121 patch is missing")
    fallback_source = FALLBACK.read_text() if FALLBACK.is_file() else ""
    if (
        "qsa_sm121_varlen_attention" not in fallback_source
        or "finite & valid" not in fallback_source
    ):
        raise SystemExit("Qwen sparse-attention fallback kernel is missing")
    for relative in (
        "managers/schedule_batch.py",
        "managers/scheduler_components/batch_result_processor.py",
        "managers/scheduler.py",
    ):
        path = SRT / relative
        if not path.is_file() or "dspark_token0_guard" not in path.read_text():
            raise SystemExit(f"Qwen token-0 cache guard is missing from {relative}")
    if not NCCL.is_file() or NCCL.read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL 2.30.7 runtime is missing or invalid")
    wrapper = Path("/opt/vonk/bin/sglang-serve")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("SGLang wrapper is missing or not executable")


if __name__ == "__main__":
    main()
