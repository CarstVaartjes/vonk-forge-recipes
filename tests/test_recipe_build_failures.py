from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
MINIMAX_DUAL = ROOT / "adapters/minimax/a3refaat-m3-b12x-dual"
SKIN_CONTEXT = ROOT / "adapters/three-d/skintokens"


def run_steps(dockerfile: Path) -> list[str]:
    return [
        line.removeprefix("RUN ")
        for line in dockerfile.read_text().replace("\\\n", " ").splitlines()
        if line.startswith("RUN ")
    ]


def test_no_adapter_builds_nccl_debian_packages() -> None:
    # dpkg-buildpackage failed every MiniMax M3 dual build (exit 2) although
    # src.build had succeeded; images use the built library directly.
    packaging = re.compile(r"pkg\.debian|debuild|dpkg-buildpackage")
    assert [
        path.relative_to(ROOT)
        for path in (ROOT / "adapters").rglob("Dockerfile")
        if packaging.search(path.read_text())
    ] == []


def test_no_adapter_builds_the_flashinfer_cubin_wheel() -> None:
    # Building flashinfer-cubin downloads ~15,600 files from NVIDIA's artifact
    # edge, which throttles a CI runner with 403s mid-burst; every MiniMax M3
    # dual build failed there. Adapters install the release wheel by hash.
    source_build = re.compile(r"\bcd \S*flashinfer-cubin\b")
    assert [
        path.relative_to(ROOT)
        for path in (ROOT / "adapters").rglob("Dockerfile")
        if source_build.search(path.read_text())
    ] == []


def test_minimax_dual_runtime_uses_the_nccl_it_builds(tmp_path: Path) -> None:
    # Run the real NCCL RUN command with network/compiler doubles. Any make
    # target besides the sm_121 src.build fails as Debian packaging did.
    commands = tmp_path / "commands"
    commands.mkdir()
    scripts = {
        "git": 'if [ "$1" = clone ]; then mkdir nccl; fi',
        "make": """case " $* " in
  *" src.build NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121 "*)
    mkdir -p build/lib
    : > build/lib/libnccl.so.2.29.7
    ln -s libnccl.so.2.29.7 build/lib/libnccl.so.2 ;;
  *) echo "debuild: fatal error: dpkg-buildpackage failed" >&2; exit 2 ;;
esac""",
    }
    for name, source in scripts.items():
        executable = commands / name
        executable.write_text(f"#!/bin/sh\nset -eu\n{source}\n")
        executable.chmod(0o755)
    dockerfile = MINIMAX_DUAL / "Dockerfile"
    step = next(step for step in run_steps(dockerfile) if "src.build" in step)
    result = subprocess.run(
        ["sh", "-ec", step],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "BUILD_JOBS": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # The step runs in WORKDIR /workspace/vllm; the runner copies that output
    # and torch's pip NCCL path is re-pointed into the copy.
    copy = re.search(
        r"^COPY --from=base /workspace/vllm/(\S+) (\S+)$",
        dockerfile.read_text(),
        re.MULTILINE,
    )
    assert copy is not None
    assert (tmp_path / copy.group(1) / "libnccl.so.2").resolve().is_file()
    verifier = load_minimax_dual_verifier()
    assert verifier.NCCL.parent == Path(copy.group(2))
    assert f"ln -s {verifier.NCCL} {verifier.PIP_NCCL}" in dockerfile.read_text()


def load_minimax_dual_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "minimax_dual_verify_runtime", MINIMAX_DUAL / "verify-runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mesh", [True, False])
def test_minimax_dual_verifier_requires_the_mesh_nccl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mesh: bool
) -> None:
    verifier = load_minimax_dual_verifier()
    built = tmp_path / "opt/nccl/lib"
    built.mkdir(parents=True)
    (built / "libnccl.so.2.29.7").touch()
    (built / "libnccl.so.2").symlink_to("libnccl.so.2.29.7")
    (tmp_path / "libnccl.so.2.29.7").touch()
    pip = tmp_path / "pip/nvidia/nccl/lib"
    pip.mkdir(parents=True)
    (pip / "libnccl.so.2").symlink_to(
        built / "libnccl.so.2" if mesh else tmp_path / "libnccl.so.2.29.7"
    )
    monkeypatch.setattr(verifier, "NCCL", built / "libnccl.so.2")
    monkeypatch.setattr(verifier, "PIP_NCCL", pip / "libnccl.so.2")
    loaded: list[str] = []

    class Library:
        def __init__(self, path: str) -> None:
            loaded.append(path)

        def ncclGetVersion(self, version: Any) -> int:
            version._obj.value = 22907
            return 0

    monkeypatch.setattr(verifier.ctypes, "CDLL", Library)
    if mesh:
        verifier.verify_nccl()
        assert loaded == [str(pip / "libnccl.so.2")]
    else:
        with pytest.raises(SystemExit, match="is not the mesh build"):
            verifier.verify_nccl()
        assert loaded == []


@pytest.mark.parametrize("replaced", [False, True])
def test_minimax_dual_verifier_requires_the_build_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replaced: bool
) -> None:
    # CI build 2026-10-27: the runner's wheel install swapped stage 1's torch
    # 2.11.0+cu130 for PyPI torch 2.10.0, so torchvision::nms did not exist.
    verifier = load_minimax_dual_verifier()
    pinned = {
        "torch": "2.11.0+cu130",
        "torchvision": "0.26.0+cu130",
        "torchaudio": "2.11.0+cu130",
        "triton": "3.6.0",
    }
    constraints = tmp_path / "torch-constraints.txt"
    constraints.write_text("".join(f"{name}=={v}\n" for name, v in pinned.items()))
    installed = {**pinned, "torch": "2.10.0"} if replaced else pinned
    monkeypatch.setattr(verifier, "TORCH_CONSTRAINTS", constraints)
    monkeypatch.setattr(verifier.importlib.metadata, "version", installed.__getitem__)
    monkeypatch.setitem(sys.modules, "torchvision", ModuleType("torchvision"))
    if replaced:
        with pytest.raises(SystemExit, match="torch stack differs"):
            verifier.verify_torch()
    else:
        verifier.verify_torch()


def test_skintokens_archive_host_and_digest_are_closed() -> None:
    recipe = json.loads((ROOT / "recipes/skintokens-pytorch-single.json").read_text())
    step = next(
        step for step in run_steps(SKIN_CONTEXT / "Dockerfile") if "curl" in step
    )
    url = re.search(r"https://\S+/SkinTokens/\S+", step)
    assert url is not None
    # GitHub's /archive/ redirects to codeload, which the original network
    # declaration omitted. Use the final host and an immutable commit directly.
    assert url.group() == (
        "https://codeload.github.com/VAST-AI-Research/SkinTokens/tar.gz/"
        "273b691d35989d71cd17ff2895fdc735097b92d1"
    )
    assert (
        urlsplit(url.group()).hostname
        in recipe["execution"]["build"]["network"]["hosts"]
    )
    assert "f886ce830f8f6ed5a3eabebb9399244812ac17a44d5b51fe8853c381a214e334" in step
    assert step.index("sha256sum --check --strict") < step.index("tar --extract")


@pytest.mark.parametrize("corrupt", [False, True])
def test_skintokens_verifies_download_before_use(tmp_path: Path, corrupt: bool) -> None:
    step = next(
        step for step in run_steps(SKIN_CONTEXT / "Dockerfile") if "curl" in step
    )
    # Execute the authored ingress pipeline with a fixture transport and digest,
    # so corrupted bytes must stop the shell before any archive is consumed.
    ingress = step[
        step.index("curl --fail") : step.index("&& mkdir -p /opt/skintokens")
    ]
    ingress = ingress.replace("/tmp/skintokens.tar.gz", str(tmp_path / "archive"))
    # Perl's SHA-256 checker accepts these same flags on both macOS and Linux.
    ingress = ingress.replace("sha256sum", "shasum -a 256")
    payload = b"pinned source archive fixture\n"
    ingress = ingress.replace(
        "f886ce830f8f6ed5a3eabebb9399244812ac17a44d5b51fe8853c381a214e334",
        hashlib.sha256(payload).hexdigest(),
    )
    (tmp_path / "payload").write_bytes(payload + (b"corruption" if corrupt else b""))
    curl = tmp_path / "curl"
    curl.write_text(
        '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --output ]; then cp "$PAYLOAD" "$2"; exit; fi\n'
        "  shift\ndone\nexit 2\n"
    )
    curl.chmod(0o755)
    result = subprocess.run(
        ["sh", "-ec", ingress + " && touch consumed"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PAYLOAD": str(tmp_path / "payload"),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode == 0) is not corrupt, result.stderr
    assert (tmp_path / "consumed").exists() is not corrupt


FIXTURES = ROOT / "tests/fixtures"


@pytest.mark.parametrize(
    ("listings", "accepted"),
    [
        (
            [
                "flashinfer-aot-sm121a-list-elf.json",
                "flashinfer-aot-host-only-list-elf.json",
            ],
            True,
        ),
        (["flashinfer-aot-host-only-list-elf.json"], False),
    ],
)
def test_minimax_dual_verifier_requires_gb10_flashinfer_aot_kernels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    listings: list[str],
    accepted: bool,
) -> None:
    # Fixtures are real `cuobjdump --list-elf` runs captured from this image's
    # AOT build (d768c14e, FLASHINFER_CUDA_ARCH_LIST=12.1a), with stdout, stderr
    # and exit code exactly as observed: the norm module (2 sm_121a cubins,
    # exit 0) and the host-only spdlog module (exit 255, "does not contain
    # device code" on stderr). The fake replays all three.
    verifier = load_minimax_dual_verifier()
    package = tmp_path / "flashinfer_jit_cache"
    for listing in listings:
        module = package / "jit_cache" / listing
        module.mkdir(parents=True)
        (module / "module.so").touch()
    spec = importlib.util.spec_from_loader("flashinfer_jit_cache", loader=None)
    assert spec is not None
    spec.submodule_search_locations = [str(package)]
    monkeypatch.setattr(verifier.importlib.util, "find_spec", lambda name: spec)

    def cuobjdump(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        run = json.loads((FIXTURES / Path(command[-1]).parent.name).read_text())
        return subprocess.CompletedProcess(
            command, run["returncode"], stdout=run["stdout"], stderr=run["stderr"]
        )

    monkeypatch.setattr(verifier.subprocess, "run", cuobjdump)
    if accepted:
        verifier.verify_flashinfer_aot()
    else:
        with pytest.raises(SystemExit, match="no sm_12x"):
            verifier.verify_flashinfer_aot()


# Spark build memory: vonk-forge control/src/vonk_control/catalog_entities.py
# @922747e8 lines 859 and 868: image_bytes = max over the recipe's role disks;
# memory_bytes = min(64 GiB, max(2 GiB, image_bytes)).
#
# Measured 2026-10-26 in the pinned stage-1 environment (OrbStack arm64, GNU
# time on every sampled compile, one at a time; tooling sampled every 2 s
# during the real `uv build` at MAX_JOBS=1). Keyed to the pinned source: a pin
# bump fails this test until the step is re-measured.
#   FlashInfer jit-cache: peak compiler gemm_sm120 (7,326 MiB); 1-6 commands
#     from each of the 44 module families of the 2,657 nvcc commands.
#   vLLM wheel: peak compiler w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu
#     (8,434 MiB); 1-8 commands per source family of the 350 nvcc commands,
#     every CUTLASS sm120, qutlass and fp4 file.
MINIMAX_DUAL_STEP_MEMORY = {
    "flashinfer-jit-cache": {
        "source": "https://github.com/flashinfer-ai/flashinfer.git",
        "commit": "d768c14e7cf5dd5df45a8a1de78ae815879f108a",
        "compiler_peak": 7326 * 1024**2,
        "tooling_peak": 595960 * 1024,
    },
    "wheels-vllm": {
        "source": "https://github.com/vllm-project/vllm.git",
        "commit": "979b56a66c969ab67655d2155ab4c6c5bed15f65",
        "compiler_peak": 8434 * 1024**2,
        "tooling_peak": 1037388 * 1024,
    },
}


@pytest.mark.parametrize("output", sorted(MINIMAX_DUAL_STEP_MEMORY))
def test_minimax_dual_compile_jobs_fit_the_spark_build_memory(output: str) -> None:
    measured = MINIMAX_DUAL_STEP_MEMORY[output]
    dockerfile = MINIMAX_DUAL / "Dockerfile"
    steps = run_steps(dockerfile)
    source = next(step for step in steps if str(measured["source"]) in step)
    assert str(measured["commit"]) in source, f"{output}: re-measure for the new pin"
    recipe = json.loads(
        (
            ROOT / "recipes/minimax-m3-w4a16-gptq-eagle3-tonyd2wild-vllm-dual.json"
        ).read_text()
    )
    image_bytes = max(
        role["resources"]["disk"]["image_bytes"] for role in recipe["topology"]["roles"]
    )
    limit = min(64 * 1024**3, max(2 * 1024**3, image_bytes))
    step = next(step for step in steps if output in step)
    jobs = re.search(r"\bMAX_JOBS=(\d+) ", step)
    assert jobs is not None, f"{output} step must set its own MAX_JOBS"
    need = int(jobs.group(1)) * int(measured["compiler_peak"]) + int(
        measured["tooling_peak"]
    )
    assert need <= limit


def test_skintokens_flash_attn_builds_one_architecture_offline(tmp_path: Path) -> None:
    # Run the authored pip commands with a recording python3. flash-attn's
    # defaults (four architectures, psutil-sized job count) ran nvcc out of
    # memory; its setup_requires would otherwise be fetched unpinned.
    step = next(
        step for step in run_steps(SKIN_CONTEXT / "Dockerfile") if "flash-attn" in step
    )
    pip = step[step.index("python3 -m pip install") : step.index("&& cuobjdump")]
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/sh\n"
        'printf \'%s|%s|%s|%s\\n\' "${MAX_JOBS-}" "${FLASH_ATTN_CUDA_ARCHS-}" '
        '"${FLASH_ATTENTION_FORCE_BUILD-}" "$*" >> "$CALLS"\n'
    )
    python.chmod(0o755)
    calls = tmp_path / "calls"
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"MAX_JOBS", "FLASH_ATTN_CUDA_ARCHS"}
    }
    result = subprocess.run(
        ["sh", "-ec", pip],
        env={
            **environment,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    invocations = [line.split("|", 3) for line in calls.read_text().splitlines()]
    build = next(i for i, call in enumerate(invocations) if "flash-attn==" in call[3])
    jobs, archs, force, arguments = invocations[build]
    assert int(jobs) > 0
    assert archs == "120"
    assert force == "TRUE"
    assert "--no-build-isolation" in arguments
    installed = " ".join(call[3] for call in invocations[:build]).split()
    for requirement in ("packaging", "psutil", "ninja"):
        assert any(word.startswith(f"{requirement}==") for word in installed)
