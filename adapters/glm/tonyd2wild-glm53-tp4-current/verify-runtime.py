"""Verify the consolidated GLM 5.3 SM121 adapter at build time."""

from __future__ import annotations

import os
from pathlib import Path


SITE = Path("/usr/local/lib/python3.12/dist-packages")
REQUIRED = (
    SITE / "vllm/platforms/cuda.py",
    SITE / "flashinfer/mla/_core.py",
    SITE / "vllm/model_executor/models/qwen3_dflash2.py",
    SITE / "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
    Path("/opt/vonk/bin/vllm"),
    Path("/opt/vonk/templates/glm53-chat-template-mm.jinja"),
)


def main() -> None:
    for path in REQUIRED:
        if not path.is_file():
            raise SystemExit(f"required GLM runtime file is missing: {path}")
    cuda_source = REQUIRED[0].read_text()
    if "return major in (9, 10)" not in cuda_source:
        raise SystemExit("GLM SM121 PDL safety gate is missing")
    if not os.access(REQUIRED[4], os.X_OK):
        raise SystemExit("vLLM wrapper is not executable")
    import flashinfer

    if not str(flashinfer.__version__).startswith("0.6.18"):
        raise SystemExit("FlashInfer 0.6.18 is required")


if __name__ == "__main__":
    main()
