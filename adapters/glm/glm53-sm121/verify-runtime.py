"""Verify the consolidated GLM 5.3 SM121 adapter at build time."""

from __future__ import annotations

import os
from pathlib import Path


SITE = Path("/usr/local/lib/python3.12/dist-packages")
REQUIRED = (
    SITE / "vllm/platforms/cuda.py",
    SITE / "flashinfer/mla/_core.py",
    Path("/opt/vonk/lib/libnccl.so.2"),
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
    if REQUIRED[2].read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL runtime is not an ELF library")
    if not os.access(REQUIRED[3], os.X_OK):
        raise SystemExit("vLLM wrapper is not executable")
    import flashinfer
    import ray

    if not str(flashinfer.__version__).startswith("0.6.18"):
        raise SystemExit("FlashInfer 0.6.18 is required")
    if ray.__version__ != "2.58.0":
        raise SystemExit("Ray 2.58.0 is required")


if __name__ == "__main__":
    main()
