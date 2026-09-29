"""Verify the applied DSpark overlay and vision port at image build time."""

from __future__ import annotations

from pathlib import Path

VLLM = Path("/opt/env/lib/python3.12/site-packages/vllm")
MARKERS = {
    VLLM / "v1/core/sched/scheduler.py": "is_prefill_chunk",
    VLLM / "v1/spec_decode/dspark.py": "shared_experts.gate_up_proj",
    VLLM / "models/deepseek_v4/nvidia/model.py": "aligner",
    VLLM / "model_executor/models/registry.py": "DeepseekV4VForConditionalGeneration",
    VLLM / "utils/torch_utils.py": "nvfp4_ds_mla",
}


def main() -> None:
    for path, marker in MARKERS.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"DSpark runtime patch is missing or wrong: {path}")
    for name in ("ds4v_vision.py", "ds4v_mm.py"):
        if not (VLLM / "models/deepseek_v4/nvidia" / name).is_file():
            raise SystemExit(f"vision port file is missing: {name}")
    if not Path("/opt/vonk/bin/vllm").is_file():
        raise SystemExit("Controller vLLM wrapper is missing")


if __name__ == "__main__":
    main()
