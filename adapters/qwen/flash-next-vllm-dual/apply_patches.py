#!/usr/bin/env python3
"""Bake the upstream Qwen3.8 Spark compatibility patches into the OCI image."""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path


BUILD = Path(__file__).resolve().parent
SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
QWEN = SITE / "models/qwen3_8_flash_next/nvidia"


def copy_original(source: Path, name: str) -> None:
    target = BUILD / name
    shutil.copyfile(source, target)


def run(script: str) -> None:
    subprocess.run(["python3", str(BUILD / script)], check=True, cwd=BUILD)


def install_checked(source: str, target: Path) -> None:
    text = (BUILD / source).read_text()
    ast.parse(text, filename=str(target))
    shutil.copyfile(BUILD / source, target)


def main() -> None:
    required = {
        "ple_layer.py": QWEN / "ple_layer.py",
        "qsa_ops_patched.py.orig": QWEN / "ops/qsa.py",
        "qsa_nvidia_patched.py.orig": QWEN / "qsa.py",
        "modelopt_patched.py.orig": SITE / "model_executor/layers/quantization/modelopt.py",
    }
    for name, source in required.items():
        if not source.is_file():
            raise SystemExit(f"pinned vLLM image is missing {source}")
        if name == "ple_layer.py":
            copy_original(source, "ple_layer_patched.py.orig")
        else:
            copy_original(source, name)

    run("patch_ple_layer.py")
    run("patch_modelopt_mxfp8.py")
    run("patch_modelopt_fp8_block_moe.py")
    run("patch_qsa_fp8_kv.py")

    install_checked("ple_layer_patched.py", QWEN / "ple_layer.py")
    install_checked("modelopt_patched.py", SITE / "model_executor/layers/quantization/modelopt.py")
    install_checked("qsa_ops_patched.py", QWEN / "ops/qsa.py")
    install_checked("qsa_nvidia_patched.py", QWEN / "qsa.py")
    print("Qwen3.8 vLLM compatibility patches baked into the image")


if __name__ == "__main__":
    main()
