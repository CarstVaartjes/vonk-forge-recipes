#!/usr/bin/env python3
"""Bake the MiaAI-Lab single-Spark Qwen3.8 Flash Next patches into the image.

Upstream start.sh extracts the original vLLM modules from the image, runs the
patch scripts in files/ and bind-mounts the results over the package at every
launch. This build step performs the same extract, patch and install once, so
the recipe image is self-contained. The patch scripts are MiaAI-Lab's from
Qwen3.8-Flash-Next-Single-DGX-Spark at commit 7d0712dc; their patch logic and
anchors are unchanged, and they were reformatted and had file handling
modernised to satisfy this repository's Ruff configuration.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent
PKG = Path("/usr/local/lib/python3.12/dist-packages/vllm")
NVIDIA = "models/qwen3_8_flash_next/nvidia"
MODELOPT = "model_executor/layers/quantization/modelopt.py"
MOE_CUTLASS = "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py"
OFFLOAD = {
    "ple_offload_layer.py": "model_executor/layers/ple_offload_layer.py",
    "connector.py": "v1/ple_offload/connector.py",
    "worker.py": "v1/ple_offload/worker.py",
    "protocol.py": "v1/ple_offload/protocol.py",
}


def extract(source: str, destination: Path) -> None:
    path = PKG / source
    if not path.is_file():
        raise SystemExit(f"pinned vLLM image is missing {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)


def run(script: str, *arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, str(BUILD / script), *arguments],
        check=True,
        cwd=BUILD,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    return result.stdout


def install(patched: Path, target: str) -> None:
    ast.parse(patched.read_text(), filename=str(patched))
    shutil.copyfile(patched, PKG / target)


def main() -> None:
    extract(f"{NVIDIA}/ple_layer.py", BUILD / "ple_layer_patched.py.orig")
    extract(MODELOPT, BUILD / "modelopt_patched.py.orig")
    extract(f"{NVIDIA}/ops/qsa.py", BUILD / "qsa_ops_patched.py.orig")
    extract(f"{NVIDIA}/qsa.py", BUILD / "qsa_nvidia_patched.py.orig")
    extract(f"{NVIDIA}/mtp.py", BUILD / "mtp_patched.py.orig")
    extract(MOE_CUTLASS, BUILD / "determinism/orig/flashinfer_cutlass_moe.py")
    for name, source in OFFLOAD.items():
        extract(source, BUILD / "ple_offload/orig" / name)
    block_drop_files = run("patch_block_drop.py", "--list").split()
    for source in block_drop_files:
        extract(source, BUILD / "block_drop/orig" / source)

    run("patch_ple_layer.py")
    run("patch_modelopt_mxfp8.py")
    run("patch_qsa_fp8_kv.py")
    run("patch_determinism.py")
    run("patch_mtp_draft_vocab.py")
    run("patch_block_drop.py")
    run("patch_ple_offload.py")

    install(BUILD / "ple_layer_patched.py", f"{NVIDIA}/ple_layer.py")
    install(BUILD / "modelopt_patched.py", MODELOPT)
    install(BUILD / "qsa_ops_patched.py", f"{NVIDIA}/ops/qsa.py")
    install(BUILD / "qsa_nvidia_patched.py", f"{NVIDIA}/qsa.py")
    install(BUILD / "mtp_patched.py", f"{NVIDIA}/mtp.py")
    install(BUILD / "determinism/flashinfer_cutlass_moe.py", MOE_CUTLASS)
    for name, target in OFFLOAD.items():
        install(BUILD / "ple_offload" / name, target)
    # Upstream mounts the block-drop files only when the pinned image lacks the
    # option; when it already has it the script writes nothing.
    for source in block_drop_files:
        patched = BUILD / "block_drop" / source
        if patched.is_file():
            install(patched, source)
    (PKG / "v1/ple_offload").mkdir(parents=True, exist_ok=True)
    print("Qwen3.8 single-Spark patches baked into the image")


if __name__ == "__main__":
    main()
