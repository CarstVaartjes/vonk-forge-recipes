"""Verify the publisher's exact GLM 5.3 SM121 DFlash2 image at build time."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


SITE = Path("/usr/local/lib/python3.12/dist-packages")
INDEXER = SITE / "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
KPOOL = SITE / "vllm/models/glm5next/nvidia/ops/kpool_compress.py"
DRAFTER = SITE / "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
WRAPPER = Path("/opt/vonk/bin/vllm")
INDEXER_SHA256 = "8a3ecfb0bab2441dd7417ed00a10d142191496149f88e5fe79fcfaea4b160980"


def main() -> None:
    for path in (INDEXER, KPOOL, DRAFTER, WRAPPER):
        if not path.is_file():
            raise SystemExit(f"required DFlash2 runtime file is missing: {path}")
    if hashlib.sha256(INDEXER.read_bytes()).hexdigest() != INDEXER_SHA256:
        raise SystemExit("exact pinned SM121 sparse-indexer replacement is missing")
    if "pid < pool_len" not in KPOOL.read_text(encoding="utf-8"):
        raise SystemExit("SM121 k-pool bounds fix is missing")
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Controller vLLM wrapper is not executable")

    from vllm.model_executor.models.registry import ModelRegistry

    if "DFlash2DraftModel" not in ModelRegistry.get_supported_archs():
        raise SystemExit("DFlash2 draft architecture is not registered")


if __name__ == "__main__":
    main()
