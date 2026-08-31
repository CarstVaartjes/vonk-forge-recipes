#!/usr/bin/env python3
"""Apply the exact Mia Vision-Exp patch sequence at build time."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SOURCE = Path("/opt/vonk-build")
PATCHES = SOURCE / "patches"
EXPECTED = {
    "apply-reasoning-default.py": "505d3345ba2a5369481896e83f94003d3e8182253f783a9852c6f45413ebaed0",
    "hotfix-encoding-dsv4-issue21.py": "c75d160245cb563d6e9a6adaee9bf7a4cd55ed5268b5ca89856977d293df9816",
    "hotfix-dsv4-issue55-tool-truncation.py": "471d1f9b69f487f6ee6c74025ec4700cf68a046afc166e455a6846c213ce418f",
    "hotfix-nvfp4-ds-mla-issue22.sh": "52de6d0cd06f571cfdbbb856bfea4a098a5118ce769de739a18b51044300772a",
    "hotfix-gb10-spin-wait.sh": "b7deed123348d78c8e7ae3f99d9107b59a798d0b3c7840b7cccaadfd8418de71",
    "hotfix-dsv4-mtp-buffer-50312.sh": "8ad604b767e09390a958cd6ffd907dd9260bb92d680b8d4d5e702ff61fb787f4",
    "hotfix-dsv4-skip-topk-49486.sh": "431eff0d51c107afacc8ddb76e34c5a57d146341bf5a0d982569e8f89fc474ed",
    "hotfix-dsv4-dense-prefill-indexer-48407.sh": "6d731f1b03b6c17275c8f0af82ee5dfa3ff9d778d25468b4edc96fbd356ffa23",
    "hotfix-dsv4-skip-empty-c128-48957.sh": "bcae8526f474f885f0af681aaa596e613fa94f8bf95847f1e71c5ea4970ccd27",
    "hotfix-dsv4-flashmla-workspace-50298.sh": "213fd93fb6c4dd70f38eefbd331f0ce08b64331feb2ff03643394857acd96078",
    "hotfix-dsv4-grammar-advance.sh": "6318c0959816156ba0015fba9d3d56e4e128acdfb778aee373d9bf227c6faaa5",
    "hotfix-vllm-empty-encoder-output.py": "e417bcdcb6d62f4790885fe5c64bef3a3015a17cea00e3901eb3e2f4b7cf35a6",
    "hotfix-dsv4-issue27-partial-prefill-concurrency.py": "1b99e7e220d027bca313ae024785a222e790e1cc62513e4c17fc0d33bda89956",
    "hotfix-dsv4-issue43-decode-fairness-and-diag.py": "84b331feb2a0c3e2f4785c94c5d99246c79bc4280e49f19c3c9783468cf7057b",
    "hotfix-dsv4-issue26-hybrid-swa-min.py": "8c76a65207d5f30b898cf5f60e39b8a59e4febb3217c34fe57f6a7fb225a3c3f",
    "hotfix-dsv4-issue133-triton-specialization.py": "64d23c25fdd40bf1d6418c217c76d90b6eed8991a26f66c92d002b4b45523b3f",
    "hotfix-vllm-issue136-xgrammar-termination.py": "f6c4690d3de7d7325d21c708a5c6b46aa7249665b8206785b5c4cd5c4e108ccb",
    "hotfix-dsv4-suppress-stops-in-reasoning.py": "618a66c58fc422ae65d0f08018fac69370657e4ead1285c8104a56f507f6279f",
    "hotfix-dsv4-vision-exp.py": "882c26ed30e1e2f611bd902bd2ee63853f4eeea3eb3ca23137b2adf8b27449e8",
}
OVERLAY_EXPECTED = {
    "__init__.py": "ab22bd0f2d77a29c7e9253c65104f07578c007debdadf09513e9998e0c22b81f",
    "apply.py": "1042d54609cc2741807166429bce78c7860932836596a1adb310f5cce3dfc0ef",
    "image_processor.py": "7424cbe1db0319371844dc6dfcd7640eb405d1c722f394eebc1ec6ff8c4687fe",
    "processor.py": "ac9b01871286bf088f4df75547e7b6384e1990209c3c472ce0e63ce8c20ae142",
    "vision.py": "e29feb76d7b7abfc5ae15fd152ded145d3c7c370030dfd35a0d96565112b3891",
}


def checked_patch(name: str) -> Path:
    path = PATCHES / name
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED[name]:
        raise SystemExit(f"patch hash mismatch: {name}")
    return path


def run(name: str, *arguments: str, environment: dict[str, str] | None = None) -> None:
    path = checked_patch(name)
    command = ["python3" if path.suffix == ".py" else "bash", str(path), *arguments]
    subprocess.run(command, check=True, env=environment)


if not ROOT.is_dir():
    raise SystemExit(f"vLLM tree is missing: {ROOT}")
encoding = SOURCE / "encoding/encoding_dsv4.py"
if hashlib.sha256(encoding.read_bytes()).hexdigest() != (
    "b4bbb74bbb11a9c8ada04daa30cc7de7dba3abba08e9ade06d38b51a3d0d1701"
):
    raise SystemExit("official encoding hash mismatch")
for name, expected in OVERLAY_EXPECTED.items():
    overlay = PATCHES / "vision_exp" / name
    if hashlib.sha256(overlay.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Vision-Exp overlay hash mismatch: {name}")
destination = ROOT / "tokenizers/deepseek_v4_encoding.py"
shutil.copyfile(encoding, destination)

run("apply-reasoning-default.py")
run("hotfix-encoding-dsv4-issue21.py", str(destination))
run("hotfix-dsv4-issue55-tool-truncation.py", str(ROOT))

shell_environment = {**os.environ, "VLLM_ROOT": str(ROOT)}
run("hotfix-nvfp4-ds-mla-issue22.sh", environment=shell_environment)
run("hotfix-gb10-spin-wait.sh", environment=shell_environment)
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
run(
    "hotfix-dsv4-vision-exp.py",
    str(PATCHES / "vision_exp"),
    str(ROOT / "models/deepseek_v4/nvidia/model.py"),
    str(destination),
    str(ROOT / "models/deepseek_v4/nvidia/dspark.py"),
)
for backup in ROOT.rglob("*.bak"):
    backup.unlink()
for cache in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache)
