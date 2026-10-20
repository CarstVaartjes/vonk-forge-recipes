"""Diffusers LoRA calls need an installed, compatible PEFT backend."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
LORA_APIS = ("load_lora_weights", "fuse_lora", "set_adapters")
SMOKE = (
    "import peft; from diffusers.utils import USE_PEFT_BACKEND; assert USE_PEFT_BACKEND"
)


def _calls_lora(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if not any(api in source for api in LORA_APIS):
        return False
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in LORA_APIS
            or isinstance(node.func, ast.Name)
            and node.func.id in LORA_APIS
        )
        for node in ast.walk(ast.parse(source))
    )


def _check_image(context: Path) -> None:
    lock_path = context / "requirements.lock"
    assert lock_path.is_file(), f"{context}: LoRA image needs a PEFT lock"
    lock = lock_path.read_text().replace("\\\n", " ")
    assert re.search(
        r"^peft==\d+\.\d+\.\d+\s+(?:--hash=sha256:[0-9a-f]{64}\s*)+$",
        lock,
        re.MULTILINE,
    ), f"{context}: pin PEFT with distribution hashes"
    dockerfile = (context / "Dockerfile").read_text().replace("\\\n", " ")
    assert "COPY requirements.lock " in dockerfile, context
    assert re.search(
        r"pip install[^&\n]*--require-hashes[^&\n]*--requirement \S*/requirements.lock",
        dockerfile,
    ), f"{context}: install the hash-checked PEFT lock"
    assert f'python3 -c "{SMOKE}"' in dockerfile, (
        f"{context}: import PEFT and check USE_PEFT_BACKEND at image build time"
    )


def test_every_lora_adapter_image_installs_hash_locked_peft() -> None:
    contexts = [
        context
        for context in sorted(ADAPTERS.glob("*/*"))
        if context.is_dir() and any(_calls_lora(p) for p in context.rglob("*.py"))
    ]
    assert contexts, "the guard must find the Qwen Image Lightning LoRA caller"
    for context in contexts:
        _check_image(context)


@pytest.mark.parametrize("api", LORA_APIS)
def test_guard_detects_each_lora_api(tmp_path: Path, api: str) -> None:
    path = tmp_path / "job.py"
    path.write_text(f"pipe.{api}(\n    'adapter'\n)\n")
    assert _calls_lora(path)
    path.write_text(f"{api}('adapter')\n")
    assert _calls_lora(path)
    path.write_text(f"# pipe.{api}('adapter')\nmessage = '{api}'\n")
    assert not _calls_lora(path)


@pytest.mark.parametrize(
    "fault", ["missing-lock", "missing-peft", "hash", "install", "smoke"]
)
def test_guard_rejects_broken_lora_images(tmp_path: Path, fault: str) -> None:
    context = ADAPTERS / "image/qwen-image-lightning-diffusers"
    for name in ("Dockerfile", "requirements.lock"):
        (tmp_path / name).write_text((context / name).read_text())
    _check_image(tmp_path)
    lock = tmp_path / "requirements.lock"
    dockerfile = tmp_path / "Dockerfile"
    if fault == "missing-lock":
        lock.unlink()
    elif fault == "missing-peft":
        lock.write_text("# PEFT omitted\n")
    elif fault == "hash":
        lock.write_text(re.sub(r"--hash=sha256:[0-9a-f]{64}", "", lock.read_text()))
    elif fault == "install":
        dockerfile.write_text(dockerfile.read_text().replace("--require-hashes", ""))
    else:
        dockerfile.write_text(dockerfile.read_text().replace(SMOKE, "import diffusers"))
    with pytest.raises(AssertionError):
        _check_image(tmp_path)
