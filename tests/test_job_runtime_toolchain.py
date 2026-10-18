"""CUDA runtime bases omit the host compiler needed by PyTorch's Triton JIT."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def python_cuda_runtimes() -> list[Path]:
    runtimes = []
    for path in sorted((ROOT / "adapters").rglob("Dockerfile")):
        source = path.read_text(encoding="utf-8")
        # Inspect the final stage: a compiler in a builder is not a runtime compiler.
        final_stage = re.split(r"^FROM ", source, flags=re.MULTILINE)[-1]
        base = final_stage.splitlines()[0]
        if (
            base.startswith("nvcr.io/nvidia/cuda:")
            and "-runtime-" in base
            and "pip install" in final_stage
        ):
            runtimes.append(path)
    return runtimes


@pytest.mark.parametrize(
    "dockerfile", python_cuda_runtimes(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_python_cuda_runtime_keeps_triton_host_toolchain(dockerfile: Path) -> None:
    # All our Python CUDA runtime images install torch (directly or from a
    # lockfile), which ships Triton. Do not rely on torch.compile being explicit:
    # attention implementations can JIT on their own, too.
    source = dockerfile.read_text(encoding="utf-8")
    final_stage = re.split(r"^FROM ", source, flags=re.MULTILINE)[-1]
    commands = final_stage.replace("\\\n", " ")
    installs = re.findall(r"apt-get install\b([^;&\n]+)", commands)
    packages = {word for command in installs for word in command.split()}
    assert "build-essential" in packages, (
        f"{dockerfile}: Triton needs a runtime C compiler and libc headers"
    )
    assert "python3-dev" in packages, (
        f"{dockerfile}: Triton needs Python.h for its launcher"
    )
    assert not re.search(
        r"apt-get (?:purge|remove).*\b(build-essential|gcc|python3-dev)\b", commands
    )
