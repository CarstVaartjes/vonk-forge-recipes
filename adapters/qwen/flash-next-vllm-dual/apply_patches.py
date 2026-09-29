#!/usr/bin/env python3
"""Bake the vLLM 0.30 Qwen4Exp overlays for the Spark Flash Next lane.

Both overlays follow MiaAI-Lab's ``start-v030.sh`` lane: the reduced MTP draft
vocabulary and the vllm#55557 FP8 KV backport (used only when the recipe
selects ``--kv-cache-dtype fp8``; BF16 KV, the lane default, is unchanged).
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent
QWEN = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia")
FP8 = BUILD / "v030_fp8kv"


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(BUILD / script), *args], check=True, cwd=BUILD)


def install_checked(source: Path, target: Path) -> None:
    ast.parse(source.read_text(), filename=str(target))
    shutil.copyfile(source, target)


def main() -> None:
    for source in (QWEN / "mtp.py", QWEN / "qsa.py", QWEN / "ops/qsa.py"):
        if not source.is_file():
            raise SystemExit(f"pinned vLLM image is missing {source}")
    shutil.copyfile(QWEN / "mtp.py", BUILD / "mtp_v030_patched.py.orig")
    (FP8 / "orig/ops").mkdir(parents=True)
    shutil.copyfile(QWEN / "qsa.py", FP8 / "orig/qsa.py")
    shutil.copyfile(QWEN / "ops/qsa.py", FP8 / "orig/ops/qsa.py")

    run("patch_mtp_draft_vocab_v030.py")
    run("patch_qsa_fp8_kv_v030.py")

    install_checked(BUILD / "mtp_v030_patched.py", QWEN / "mtp.py")
    install_checked(FP8 / "qsa.py", QWEN / "qsa.py")
    install_checked(FP8 / "ops/qsa.py", QWEN / "ops/qsa.py")
    shutil.copyfile(BUILD / "draft_vocab_en_code_47k.txt", "/etc/vllm-draft-vocab.txt")
    print("Qwen3.8 vLLM 0.30 lane overlays baked into the image")


if __name__ == "__main__":
    main()
