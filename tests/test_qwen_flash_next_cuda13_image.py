"""The tonyd2wild Qwen3.8 Flash Next image runs the CUDA 13 build of upstream's vLLM.

The CUDA 12.9 build of the same commit carries its SM12x kernels as sm_120a only,
which GB10 (sm_121) cannot load ("no kernel image is available"). The adapter's
build-time scan rejects such a fatbin; the entry flags below are the ones measured
in vLLM's 8a728663 cu129 (sm_120a: 0x1100011) and CUDA 13 (sm_120f: 0x1208011)
aarch64 wheels.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

from documents import recipe

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/qwen/flash-next-tonyd2wild-vllm"
CUDA13_BASE = "5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
WHEEL_SHA256 = "5132d531c0ef9cee849f8a2f3262da2ad0d62dd7a404e1108092cd08a3bc00a3"

spec = importlib.util.spec_from_file_location(
    "verify_runtime", ADAPTER / "verify-runtime.py"
)
assert spec and spec.loader
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

PLAIN = 0x1000011
SM120A = 0x1100011
SM120F = 0x1208011


def fatbin(*entries: tuple[int, int, int]) -> bytes:
    body = b""
    for kind, sm, flags in entries:
        header = struct.pack("<HHIQ", kind, 0x101, 64, 8)
        header += struct.pack("<IIHHI", 0, 0, 8, 1, sm)
        header += struct.pack("<IIQQQ", 0, 0, flags, 0, 8)
        body += header + b"\0" * 8
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def test_header_layout_matches_measured_wheel_entries() -> None:
    assert verify.fatbin_entries(fatbin((2, 120, SM120A), (1, 80, 0x11)), 0) == [
        (2, 120, SM120A),
        (1, 80, 0x11),
    ]


def test_sm120a_only_fatbin_is_stranded_on_gb10() -> None:
    data = b"pad" + fatbin((2, 90, 0x100011), (2, 120, SM120A))
    assert verify.stranded_fatbins(data) == [3]


def test_cuda12_plain_sm120_and_arch_specific_ptx_are_stranded() -> None:
    assert verify.stranded_fatbins(
        fatbin((2, 80, 0x11), (2, 120, SM120A), (1, 120, SM120A))
    )
    assert verify.stranded_fatbins(fatbin((2, 100, SM120A), (2, 120, SM120A)))


def test_family_plain_sm121_and_ptx_images_load_on_gb10() -> None:
    for entries in (
        ((2, 120, SM120F),),
        ((2, 120, PLAIN),),
        ((2, 121, SM120A),),
        ((2, 120, SM120A), (1, 120, 0x1000011)),
        ((2, 90, 0x100011), (2, 120, SM120A), (2, 120, SM120F)),
    ):
        assert verify.stranded_fatbins(fatbin(*entries)) == [], entries


def test_family_ptx_only_loads_inside_its_family() -> None:
    assert verify.stranded_fatbins(fatbin((1, 100, 0x1200011), (2, 120, SM120A)))
    assert not verify.stranded_fatbins(fatbin((1, 120, 0x1200011), (2, 120, SM120A)))


def test_libraries_without_sm12x_code_are_not_judged() -> None:
    assert verify.stranded_fatbins(fatbin((2, 90, 0x100011), (2, 100, SM120A))) == []
    assert verify.stranded_fatbins(b"no fatbin here") == []


def test_every_fatbin_in_a_library_is_checked() -> None:
    data = fatbin((2, 120, SM120F)) + b"\0" * 5 + fatbin((2, 120, SM120A))
    assert verify.stranded_fatbins(data) == [len(fatbin((2, 120, SM120F))) + 5]


def test_dockerfile_installs_the_locked_cuda13_build_and_scans_kernels() -> None:
    dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
    assert (
        f"FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:{CUDA13_BASE}"
        in dockerfile
    )
    assert (
        "pip install --ignore-installed --no-deps --require-hashes"
        in dockerfile.replace("python3 -m ", "")
    )
    assert (
        "import torch, vllm._C_stable_libtorch, vllm.entrypoints.openai.api_server"
        in dockerfile
    )
    assert "verify-runtime.py kernels" in dockerfile
    lock = (ADAPTER / "requirements.lock").read_text(encoding="utf-8")
    assert f"--hash=sha256:{WHEEL_SHA256}" in lock
    assert (
        "8a728663c1c3eeace834a95f5654fa653cc1998c/vllm-0.28.1rc1.dev388%2Bg8a728663c-cp38"
        in lock
    )
    for pin in (
        "torch==2.13.0+cu130",
        "torchvision==0.28.0+cu130",
        "torchaudio==2.11.0+cu130",
    ):
        assert pin in lock
    assert "flashinfer-jit-cache==0.6.18+cu130" in lock
    assert "nvidia-nccl-cu13==2.30.7" in lock
    assert "nvidia-nccl-cu13==2.29.7" not in lock


def test_recipes_declare_the_cuda13_base_and_its_hosts() -> None:
    for topology in ("single", "dual", "four"):
        build = recipe(
            f"qwen3-8-flash-next-nvfp4-tonyd2wild-vllm-{topology}"
        ).execution.build
        assert build.base_image.digest == CUDA13_BASE
        assert build.base_image.repository == "nvcr.io/nvidia/cuda"
        assert {"wheels.vllm.ai", "download.pytorch.org", "flashinfer.ai"} <= set(
            build.network.hosts
        )
