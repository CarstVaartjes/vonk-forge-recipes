"""A baked cache that a workload copies at start is readable and never fatal.

The platform runs every workload as its own runtime user, never the image's
(spark-3542: uid 128, gid 127). ``cp -a`` keeps the upstream owner's private
modes, so a seed copied from ``/opt/vonk/prewarm`` failed with ``Permission
denied`` and the workload never started (230 failed starts of
ornith-1-5-35b-a3b-nvfp4-w4a16-r0b0tlab-sglang-single).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
PREWARM = "/opt/vonk/prewarm"
#: Launchers that copy a baked cache at start, each with a behavioural test here.
SEEDING_LAUNCHERS = {"adapters/nvidia/r0b0tlab-ornith-sglang-single/sglang-serve"}


def _own_sources() -> list[Path]:
    return [
        path
        for path in ADAPTERS.rglob("*")
        if path.is_file()
        and "vendor" not in path.relative_to(ADAPTERS).parts
        and path.stat().st_size < 1_000_000
    ]


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def test_every_image_that_bakes_a_prewarm_cache_makes_it_readable() -> None:
    dockerfiles = [path for path in _own_sources() if path.name == "Dockerfile"]
    baking = [path for path in dockerfiles if PREWARM in _text(path)]
    assert baking, "the guard no longer sees any prewarm image"
    for path in baking:
        assert re.search(
            rf"chmod\s+-R\s+a\+rX\s+{re.escape(PREWARM)}\b", _text(path)
        ), f"{path.relative_to(ROOT)}: {PREWARM} must be readable by the runtime user"


def test_every_launcher_that_copies_a_cache_at_start_is_tested_here() -> None:
    copying = {
        path.relative_to(ROOT).as_posix()
        for path in _own_sources()
        if "copytree(" in _text(path)
    }
    assert copying == SEEDING_LAUNCHERS


def _launcher(relative: str):
    path = ROOT / relative
    loader = importlib.machinery.SourceFileLoader("launcher_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", sorted(SEEDING_LAUNCHERS))
def test_a_readable_seed_is_copied_once(tmp_path: Path, relative: str) -> None:
    launcher = _launcher(relative)
    source = tmp_path / "prewarm" / "sglang" / "flashinfer"
    source.mkdir(parents=True)
    (source / "rank_tp0_pp0_dp0.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "cache" / "sglang"

    launcher.seed(tmp_path / "prewarm" / "sglang", target)

    assert (target / "flashinfer" / "rank_tp0_pp0_dp0.json").read_text() == "{}"
    assert (target / ".vonk-prewarmed").is_file()


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads any file; the refusal needs another user"
)
@pytest.mark.parametrize("relative", sorted(SEEDING_LAUNCHERS))
def test_an_unreadable_seed_starts_without_it(
    tmp_path: Path, relative: str, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = _launcher(relative)
    source = tmp_path / "prewarm" / "sglang"
    source.mkdir(parents=True)
    private = source / "rank_tp0_pp0_dp0.json"
    private.write_text("{}", encoding="utf-8")
    private.chmod(0)  # another user cannot read the owner's 0600 file
    target = tmp_path / "cache" / "sglang"
    try:
        launcher.seed(source, target)
    finally:
        private.chmod(0o600)

    assert not (target / ".vonk-prewarmed").exists()
    assert "starting without it" in capsys.readouterr().err
