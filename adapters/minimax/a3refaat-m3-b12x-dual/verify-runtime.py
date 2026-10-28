"""Verify the MiniMax-M3 b12x stack and launch wrapper at image build time."""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import os
import re
import subprocess
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKERS = {
    SITE / "models/minimax_m3/common/b12x_backend.py": "b12x.vllm.minimax_m3.backend",
}
WRAPPER = Path("/opt/vonk/bin/vllm")
NCCL = Path("/opt/nccl/lib/libnccl.so.2")
PIP_NCCL = Path("/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2")
TORCH_CONSTRAINTS = Path("/opt/vonk/torch-constraints.txt")
TORCH_DISTRIBUTIONS = ("torch", "torchvision", "torchaudio", "triton")


def verify_torch() -> None:
    # vLLM and FlashInfer were compiled against stage 1's torch (its freeze is the constraints file); a resolution that
    # replaces it leaves torchvision's ops and the compiled extensions unloadable.
    pinned = dict(line.split("==", 1) for line in TORCH_CONSTRAINTS.read_text(encoding="utf-8").split())
    installed = {name: importlib.metadata.version(name) for name in TORCH_DISTRIBUTIONS}
    if pinned != installed:
        raise SystemExit(f"torch stack differs from the one the wheels were built against: {installed} != {pinned}")
    import torchvision  # noqa: F401  registers torchvision::nms against the installed torch

    print("torch stack:", installed)


def verify_nccl() -> None:
    # torch loads NCCL through the pip package's path: it must resolve to the mesh build, which must load without a GPU.
    if PIP_NCCL.resolve() != NCCL.resolve():
        raise SystemExit(f"torch's NCCL {PIP_NCCL} is not the mesh build {NCCL}")
    version = ctypes.c_int()
    if ctypes.CDLL(str(PIP_NCCL)).ncclGetVersion(ctypes.byref(version)) != 0:
        raise SystemExit(f"ncclGetVersion failed for {PIP_NCCL}")
    print("NCCL mesh build OK:", version.value)


def verify_flashinfer_cubin_version() -> None:
    # flashinfer refuses to import when flashinfer-cubin's version differs from its own (flashinfer/jit/env.py);
    # the cubin wheel is the PyPI release, the python wheel is built from the same commit.
    versions = {name: importlib.metadata.version(name) for name in ("flashinfer-python", "flashinfer-cubin")}
    if len(set(versions.values())) != 1:
        raise SystemExit(f"FlashInfer wheel versions differ: {versions}")
    print("FlashInfer wheels:", versions)


def verify_flashinfer_aot() -> None:
    # The AOT kernel cache must hold GB10 (sm_12x) cubins; one cuobjdump per AOT .so, once per image build.
    spec = importlib.util.find_spec("flashinfer_jit_cache")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("flashinfer_jit_cache is not installed")
    root = Path(next(iter(spec.submodule_search_locations))) / "jit_cache"
    archs: set[str] = set()
    for library in sorted(root.rglob("*.so")):
        result = subprocess.run(["cuobjdump", "--list-elf", str(library)], capture_output=True, text=True, check=False)
        if result.returncode != 0 and "does not contain device code" not in result.stdout + result.stderr:
            raise SystemExit(f"cuobjdump failed for {library}: {result.stderr.strip()}")
        archs.update(re.findall(r"\.(sm_\d+[a-z]?)\.cubin", result.stdout))
    if not any(arch.startswith("sm_12") for arch in archs):
        raise SystemExit(f"FlashInfer AOT kernels hold no sm_12x (GB10) cubins; found {sorted(archs)} in {root}")
    print("FlashInfer AOT kernels:", sorted(archs))


def main() -> None:
    verify_torch()
    verify_nccl()
    verify_flashinfer_cubin_version()
    verify_flashinfer_aot()
    # Upstream Dockerfile.deploy smoke imports, run after the mods so a broken layer fails the build, not the launch.
    import importlib.util

    import b12x
    from b12x.integration.attention import (  # noqa: F401
        B12XPagedAttentionScratchCaps,
        clear_attention_caches,
        paged_attention_forward,
        plan_paged_attention_scratch,
    )

    for module in (
        "b12x.vllm.minimax_m3.backend",
        "b12x.vllm.minimax_m3.indexer",
        "b12x.vllm.minimax_m3.msa_attn",
        "b12x.vllm.minimax_m3.triton_prewarm",
    ):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"missing {module}")
    for rel in (
        "models/minimax_m3/common/sparse_attention.py",
        "models/minimax_m3/common/indexer.py",
        "v1/attention/backends/fa_utils.py",
        "v1/spec_decode/llm_base_proposer.py",
    ):
        if not (SITE / rel).exists():
            raise SystemExit(f"base image missing vllm/{rel}: wrong vLLM ref?")
    print("b12x baked OK:", getattr(b12x, "__version__", "unversioned"))
    for path, marker in MARKERS.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"MiniMax-M3 b12x mod is missing or wrong: {path}")
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Controller vLLM wrapper is not executable")


if __name__ == "__main__":
    main()
