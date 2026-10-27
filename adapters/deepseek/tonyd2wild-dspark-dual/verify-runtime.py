"""Verify the applied DSpark overlay, and the vision port or its absence, at image build time.

Without arguments the image must carry the Vision-Exp port; with --text it must not,
because the port makes DeepseekV4ForCausalLM multimodal for every checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

VLLM = Path("/opt/env/lib/python3.12/site-packages/vllm")
MODEL_DIR = VLLM / "models/deepseek_v4/nvidia"
MARKERS = {
    VLLM / "v1/core/sched/scheduler.py": "is_prefill_chunk",
    VLLM / "v1/spec_decode/dspark.py": "shared_experts.gate_up_proj",
    VLLM / "utils/torch_utils.py": "nvfp4_ds_mla",
}
VISION_MARKERS = {
    MODEL_DIR / "model.py": "aligner",
    VLLM / "model_executor/models/registry.py": "DeepseekV4VForConditionalGeneration",
}
VISION_FILES = ("ds4v_vision.py", "ds4v_mm.py")


def _carries(path: Path, marker: str) -> bool:
    return path.is_file() and marker in path.read_text(encoding="utf-8")


def main(arguments: list[str]) -> None:
    if arguments not in ([], ["--text"]):
        raise SystemExit("usage: verify-runtime.py [--text]")
    vision = not arguments
    for path, marker in MARKERS.items():
        if not _carries(path, marker):
            raise SystemExit(f"DSpark runtime patch is missing or wrong: {path}")
    for path, marker in VISION_MARKERS.items():
        if _carries(path, marker) != vision:
            raise SystemExit(
                f"vision port patch presence is wrong for this build: {path}"
            )
    for name in VISION_FILES:
        if (MODEL_DIR / name).is_file() != vision:
            raise SystemExit(
                f"vision port file presence is wrong for this build: {name}"
            )
    if not Path("/opt/vonk/bin/vllm").is_file():
        raise SystemExit("Controller vLLM wrapper is missing")


if __name__ == "__main__":
    main(sys.argv[1:])
