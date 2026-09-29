"""Verify the MiniMax-M3 b12x stack and launch wrapper at image build time."""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKERS = {
    SITE / "models/minimax_m3/common/b12x_backend.py": "b12x.vllm.minimax_m3.backend",
}
WRAPPER = Path("/opt/vonk/bin/vllm")


def main() -> None:
    # Upstream Dockerfile.deploy smoke imports, run after the mods so a broken layer fails the build, not the launch.
    import importlib.util

    import b12x
    from b12x.integration.attention import (  # noqa: F401
        B12XPagedAttentionScratchCaps,
        clear_attention_caches,
        paged_attention_forward,
        plan_paged_attention_scratch,
    )

    for module in (
        "b12x.vllm.minimax_m3.backend",
        "b12x.vllm.minimax_m3.indexer",
        "b12x.vllm.minimax_m3.msa_attn",
        "b12x.vllm.minimax_m3.triton_prewarm",
    ):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"missing {module}")
    for rel in (
        "models/minimax_m3/common/sparse_attention.py",
        "models/minimax_m3/common/indexer.py",
        "v1/attention/backends/fa_utils.py",
        "v1/spec_decode/llm_base_proposer.py",
    ):
        if not (SITE / rel).exists():
            raise SystemExit(f"base image missing vllm/{rel}: wrong vLLM ref?")
    print("b12x baked OK:", getattr(b12x, "__version__", "unversioned"))
    for path, marker in MARKERS.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"MiniMax-M3 b12x mod is missing or wrong: {path}")
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Controller vLLM wrapper is not executable")


if __name__ == "__main__":
    main()
