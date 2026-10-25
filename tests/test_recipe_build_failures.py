from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
NCCL_DOCKERFILES = tuple(
    path
    for path in (ROOT / "adapters").rglob("Dockerfile")
    if "make pkg.debian.build" in path.read_text()
)
SKIN_CONTEXT = ROOT / "adapters/three-d/skintokens"


def run_steps(dockerfile: Path) -> list[str]:
    return [
        line.removeprefix("RUN ")
        for line in dockerfile.read_text().replace("\\\n", " ").splitlines()
        if line.startswith("RUN ")
    ]


@pytest.mark.parametrize("dockerfile", NCCL_DOCKERFILES)
def test_nccl_packaging_normalizes_generated_install_lists(dockerfile: Path) -> None:
    step = next(step for step in run_steps(dockerfile) if "pkg.debian.build" in step)
    assert (
        step.index("make -C pkg/debian prep")
        < step.index("chmod 0644 build/debian/*.install")
        < step.index("make pkg.debian.build")
    )
    assert 'NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"' in step


@pytest.mark.parametrize("dockerfile", NCCL_DOCKERFILES)
def test_nccl_build_handles_executable_packaging_data(
    dockerfile: Path, tmp_path: Path
) -> None:
    # Exercise the real RUN command with compiler/network/package managers
    # replaced. Inject the observed executable metadata at src.build; the
    # packaging double rejects it just as dh_install did on the Sparks.
    commands = tmp_path / "commands"
    commands.mkdir()
    scripts = {
        "git": 'if [ "$1" = clone ]; then mkdir nccl; fi',
        "make": """case " $* " in
  *" src.build "*)
    mkdir -p build/debian
    for name in libnccl2 libnccl-dev; do
      printf 'lib/libnccl.so.2 /usr/lib/aarch64-linux-gnu\n' > "build/debian/$name.install"
      chmod 0755 "build/debian/$name.install"
    done ;;
  *" pkg.debian.build "*)
    for file in build/debian/*.install; do
      if [ -x "$file" ]; then echo 'executable config: exit 132' >&2; exit 132; fi
    done ;;
esac""",
        "apt": ":",
    }
    for name, source in scripts.items():
        executable = commands / name
        executable.write_text(f"#!/bin/sh\nset -eu\n{source}\n")
        executable.chmod(0o755)
    step = next(step for step in run_steps(dockerfile) if "pkg.debian.build" in step)
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
    for manifest in (tmp_path / "nccl/build/debian").glob("*.install"):
        assert manifest.read_text() == "lib/libnccl.so.2 /usr/lib/aarch64-linux-gnu\n"
        assert manifest.stat().st_mode & 0o111 == 0


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
