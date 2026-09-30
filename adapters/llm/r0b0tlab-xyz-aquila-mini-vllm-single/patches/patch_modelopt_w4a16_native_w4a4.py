#!/usr/bin/env python3
"""Patch ModelOpt mixed linear routing for the pinned NVIDIA checkpoint.

The target checkpoint labels eligible ordinary linear layers W4A16_NVFP4 but
ships calibrated input scales for W4A4 execution. This patch changes only the
ordinary ``LinearBase``/``ParallelLMHead`` dispatcher branch. Routed experts
remain on the independent native MoE backend.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

MODULE = "vllm.model_executor.layers.quantization.modelopt"
MARKER = "R0B0TLAB_NATIVE_W4A4_FROM_W4A16"
OLD = '''            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4W4A16LinearMethod(self.w4a16_nvfp4_config)
'''
NEW = f'''            if quant_algo == "W4A16_NVFP4":
                logger.warning_once(
                    "{MARKER}: routing calibrated W4A16-labelled linear "
                    "layers through native NVFP4 W4A4; checkpoint input_scale "
                    "tensors are required and consumed."
                )
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
'''


def patch_text(text: str) -> tuple[str, str]:
    if MARKER in text:
        if OLD in text:
            raise ValueError("marker and unpatched routing block coexist")
        return text, "already-patched"
    if text.count(OLD) != 1:
        raise ValueError("expected exactly one W4A16 ordinary-linear routing block")
    patched = text.replace(OLD, NEW)
    if MARKER not in patched or OLD in patched:
        raise ValueError("post-patch verification failed")
    return patched, "patched"


def locate_module_path() -> Path:
    spec = importlib.util.find_spec(MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {MODULE}")
    return Path(spec.origin)


def patch_file(path: Path) -> str:
    text = path.read_text()
    patched, status = patch_text(text)
    if status == "patched":
        path.write_text(patched)
    current = path.read_text()
    check, check_status = patch_text(current)
    if check != current or check_status != "already-patched":
        raise RuntimeError("on-disk post-patch verification failed")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, help="Explicit modelopt.py path")
    args = parser.parse_args()
    path = args.path or locate_module_path()
    status = patch_file(path)
    print(f"{status}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
