#!/usr/bin/env python3
"""Strict runtime audit for the Agents-A1 NVFP4 SM121 container."""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import site
import sys

import torch
import vllm


def main() -> int:
    report = {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "env": {
            "VLLM_NVFP4_GEMM_BACKEND": os.environ.get("VLLM_NVFP4_GEMM_BACKEND"),
            "VLLM_FP8_MOE_BACKEND": os.environ.get("VLLM_FP8_MOE_BACKEND"),
            "CUTE_DSL_ARCH": os.environ.get("CUTE_DSL_ARCH"),
        },
        "imports": {},
        "fp4_support": {},
        "fallback_artifacts": {},
    }

    for mod in ["vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C"]:
        try:
            importlib.import_module(mod)
            report["imports"][mod] = "ok"
        except Exception as exc:
            report["imports"][mod] = f"FAIL: {exc!r}"

    try:
        report["fp4_support"]["121"] = bool(torch.ops._C.cutlass_scaled_mm_supports_fp4(121))
        report["fp4_support"]["120"] = bool(torch.ops._C.cutlass_scaled_mm_supports_fp4(120))
    except Exception as exc:
        report["fp4_support"]["error"] = repr(exc)

    roots = [pathlib.Path("/workspace/vllm"), pathlib.Path("/opt/vllm-src"), *map(pathlib.Path, site.getsitepackages())]
    for pat in ["*_vllm_fa2_C*.so", "*_vllm_fa3_C*.so", "*marlin*.so"]:
        hits = []
        for root in roots:
            if root.exists():
                hits.extend(str(p) for p in root.rglob(pat))
        report["fallback_artifacts"][pat] = hits

    print(json.dumps(report, indent=2, sort_keys=True))
    failures = []
    if not all(v == "ok" for v in report["imports"].values()):
        failures.append("extension import failure")
    if report["fp4_support"].get("121") is not True or report["fp4_support"].get("120") is not True:
        failures.append("cutlass FP4 support check failed")
    for pat, hits in report["fallback_artifacts"].items():
        if hits:
            failures.append(f"forbidden packaged artifacts for {pat}: {hits[:5]}")
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) != (12, 1):
        failures.append(f"expected SM121, got {torch.cuda.get_device_capability(0)}")
    if failures:
        print("AUDIT FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("AUDIT PASS: SM121 native NVFP4 runtime gates satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
