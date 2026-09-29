"""Fail the adapter build unless the image carries the matched GLM 5.3 kernel set."""

from __future__ import annotations

import os
import re
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
NCCL = Path("/opt/vonk/lib/libnccl.so.2")
VLLM = Path("/usr/local/bin/vllm")
INDEXER = SITE / "v1/attention/backends/mla/indexer.py"


def main() -> None:
    for path in (NCCL, VLLM):
        if not path.is_file():
            raise SystemExit(f"required GLM runtime file is missing: {path}")
    if NCCL.read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL runtime is not an ELF library")
    if not os.access(VLLM, os.X_OK):
        raise SystemExit("vLLM executable is not executable")
    deepseek = (SITE / "model_executor/models/deepseek_v2.py").read_text("utf-8")
    sparse = (SITE / "model_executor/layers/sparse_attn_indexer.py").read_text("utf-8")
    if "GlmMoeDsaForCausalLM" not in deepseek:
        raise SystemExit("deepseek_v2.py overlay does not define GlmMoeDsaForCausalLM")
    if "fused_indexer_q_rope_quant" in deepseek and not re.search(
        r"def\s+fused_indexer_q_rope_quant", sparse
    ):
        raise SystemExit("deepseek_v2.py and sparse_attn_indexer.py overlays are skewed")
    if ") + 1  # MTP spec tokens" not in INDEXER.read_text("utf-8"):
        raise SystemExit("indexer MTP-overhang patch is not applied")


if __name__ == "__main__":
    main()
