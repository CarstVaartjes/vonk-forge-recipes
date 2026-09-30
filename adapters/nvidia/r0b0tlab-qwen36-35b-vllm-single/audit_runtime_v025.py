#!/usr/bin/env python3
"""Fail-closed vLLM v0.25 runtime audit for NVIDIA Qwen on GB10/SM121."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_PACKAGES = {
    "vllm": "0.25.0",
    "flashinfer-python": "0.6.13",
    "nvidia-cutlass-dsl": "4.5.2",
    "nvidia-nccl-cu13": "2.28.9",
}
VLLM_RELEASE_COMMIT = "702f4814fe54fabff350d43cb753ae3e47c0c276"
NATIVE_MARKER = "R0B0TLAB_NATIVE_W4A4_FROM_W4A16"


def source_for_module(name: str) -> tuple[str, str]:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate module {name}")
    path = Path(spec.origin)
    return str(path), path.read_text(errors="replace")


def main() -> int:
    import torch
    import vllm

    failures: list[str] = []
    packages: dict[str, str] = {}
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            actual = md.version(name)
        except md.PackageNotFoundError:
            failures.append(f"missing package: {name}")
            continue
        packages[name] = actual
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")

    try:
        stale_jit_cache = md.version("flashinfer-jit-cache")
    except md.PackageNotFoundError:
        stale_jit_cache = None
    else:
        failures.append(f"stale flashinfer-jit-cache must be absent, found {stale_jit_cache}")

    if vllm.__version__ != "0.25.0":
        failures.append(f"vllm.__version__: expected 0.25.0, got {vllm.__version__}")
    if torch.__version__ != "2.11.0+cu130":
        failures.append(f"torch: expected 2.11.0+cu130, got {torch.__version__}")
    if torch.version.cuda != "13.0":
        failures.append(f"torch CUDA runtime: expected 13.0, got {torch.version.cuda}")

    try:
        nvcc = subprocess.run(
            ["/usr/local/cuda-13.0/bin/nvcc", "--version"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except Exception as exc:
        nvcc = ""
        failures.append(f"nvcc check failed: {exc}")
    if "release 13.0" not in nvcc:
        failures.append("active CUDA compiler is not release 13.0")

    imports: dict[str, str] = {}
    for module in (
        "torchcodec",
        "flashinfer",
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.qwen3_5_mtp",
        "vllm.v1.attention.backends.flashinfer",
    ):
        try:
            importlib.import_module(module)
            imports[module] = "PASS"
        except Exception as exc:
            imports[module] = f"FAIL: {type(exc).__name__}: {exc}"
            failures.append(f"module import failed: {module}")

    try:
        modelopt_path, modelopt_source = source_for_module(
            "vllm.model_executor.layers.quantization.modelopt"
        )
    except Exception as exc:
        modelopt_path, modelopt_source = "", ""
        failures.append(f"ModelOpt source unavailable: {exc}")
    if NATIVE_MARKER not in modelopt_source:
        failures.append("checkpoint-specific native W4A4 reroute marker is absent")

    gpu: dict[str, Any] | None = None
    if not torch.cuda.is_available():
        failures.append("CUDA GPU is unavailable")
    else:
        capability = torch.cuda.get_device_capability()
        gpu = {"name": torch.cuda.get_device_name(), "capability": capability}
        if capability != (12, 1):
            failures.append(f"GPU capability: expected (12, 1), got {capability}")
        try:
            fp4_supported = bool(torch.ops._C.cutlass_scaled_mm_supports_fp4(121))
        except Exception as exc:
            fp4_supported = False
            failures.append(f"SM121 CUTLASS FP4 probe failed: {exc}")
        gpu["cutlass_fp4_supported"] = fp4_supported
        if not fp4_supported:
            failures.append("SM121 CUTLASS FP4 support probe returned false")
        sample = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        product = sample @ sample
        torch.cuda.synchronize()
        gpu["bf16_matmul_finite"] = bool(torch.isfinite(product).all().item())
        if not gpu["bf16_matmul_finite"]:
            failures.append("CUDA BF16 matmul produced non-finite output")

    report = {
        "status": "FAIL" if failures else "PASS",
        "vllm_api_version": vllm.__version__,
        "vllm_release_commit": VLLM_RELEASE_COMMIT,
        "packages": packages,
        "flashinfer_jit_cache": stale_jit_cache,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "imports": imports,
        "modelopt_path": modelopt_path,
        "native_w4a4_marker": NATIVE_MARKER in modelopt_source,
        "gpu": gpu,
        "failures": failures,
        "provenance": "r0b0tlab-qwen36-nvidia-v025-mtp",
    }
    print(json.dumps(report, indent=2, default=list))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
