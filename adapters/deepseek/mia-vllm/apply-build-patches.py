#!/usr/bin/env python3
"""Apply the exact Mia patch sequence to the installed vLLM tree at build time."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SOURCE = Path("/opt/vonk-build")
PATCHES = SOURCE / "patches"


def checked_patch(name: str) -> Path:
    path = PATCHES / name
    if not path.is_file():
        raise SystemExit(f"patch is missing: {name}")
    return path


def run(name: str, *arguments: str, environment: dict[str, str] | None = None) -> None:
    path = checked_patch(name)
    command = ["python3" if path.suffix == ".py" else "bash", str(path), *arguments]
    subprocess.run(command, check=True, env=environment)


if not ROOT.is_dir():
    raise SystemExit(f"vLLM tree is missing: {ROOT}")
encoding = SOURCE / "encoding/encoding_dsv4.py"
# The upstream tokenizer encoding is the one content this build does not
# author: the digest proves the vLLM image still carries the file the
# patches were written against. The patch sequence itself needs no digest,
# because it ships in the same commit as this script and the vLLM tree is
# already fixed by the image digest in the Dockerfile.
if hashlib.sha256(encoding.read_bytes()).hexdigest() != (
    "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"
):
    raise SystemExit("official encoding hash mismatch")
destination = ROOT / "tokenizers/deepseek_v4_encoding.py"
shutil.copyfile(encoding, destination)

run("apply-reasoning-default.py")
run("hotfix-encoding-dsv4-issue21.py", str(destination))
run("hotfix-dsv4-issue55-tool-truncation.py", str(ROOT))

shell_environment = {**os.environ, "VLLM_ROOT": str(ROOT)}
run("hotfix-nvfp4-ds-mla-issue22.sh", environment=shell_environment)
run("hotfix-gb10-spin-wait.sh", environment=shell_environment)
run("hotfix-vllm-issue117-shm-ring-buffer.py")
for patch in (
    "hotfix-dsv4-mtp-buffer-50312.sh",
    "hotfix-dsv4-skip-topk-49486.sh",
    "hotfix-dsv4-dense-prefill-indexer-48407.sh",
    "hotfix-dsv4-skip-empty-c128-48957.sh",
    "hotfix-dsv4-flashmla-workspace-50298.sh",
    "hotfix-dsv4-grammar-advance.sh",
):
    run(patch, environment=shell_environment)

run("hotfix-vllm-empty-encoder-output.py")
run("hotfix-dsv4-issue27-partial-prefill-concurrency.py")
run("hotfix-dsv4-issue43-decode-fairness-and-diag.py")
run("hotfix-dsv4-issue26-hybrid-swa-min.py")
run("hotfix-dsv4-issue133-triton-specialization.py")
run("hotfix-vllm-issue136-xgrammar-termination.py")
run(
    "hotfix-dsv4-suppress-stops-in-reasoning.py",
    str(ROOT / "v1/engine/detokenizer.py"),
)
for backup in ROOT.rglob("*.bak"):
    backup.unlink()
for cache in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache)
