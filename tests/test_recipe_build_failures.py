from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
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
