"""An adapter line that assigns its own NCCL_IB_HCA fails; pass-through and old lines pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest
from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_nccl_ib_hca", str(ROOT / "tools/check-nccl-ib-hca")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)


@pytest.mark.parametrize(
    "code",
    [
        "  -e NCCL_IB_HCA=$HCA \\",
        '  -e NCCL_IB_HCA="rocep1s0f0"',
        "export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1",
        "NCCL_IB_HCA: roceP2p1s0f0",
        '  "NCCL_IB_HCA": "rocep1s0f0",',
        'os.environ["NCCL_IB_HCA"] = "=rocep1s0f1:1"',
        'env["NCCL_IB_HCA"] = environ["NCCL_IB_HCA"].split(",")[0]',
        "DOCKER_ENV=(-e NCCL_IB_HCA='=rocep1s0f0:1')",
        "export NCCL_IB_HCA=",
    ],
)
def test_assigning_a_value_of_its_own_is_a_hit(code: str) -> None:
    assert guard.scan(code + "\n", {1}) == [1]


@pytest.mark.parametrize(
    "code",
    [
        "export NCCL_IB_HCA=$NCCL_IB_HCA",
        'export NCCL_IB_HCA="${NCCL_IB_HCA}"',
        'NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"',
        "  -e NCCL_IB_HCA=$NCCL_IB_HCA \\",
        'os.environ["NCCL_IB_HCA"] = os.environ["NCCL_IB_HCA"]',
        '"NCCL_IB_HCA": environ["NCCL_IB_HCA"],',
        "  -e NCCL_IB_HCA",
        '    "NCCL_IB_HCA",',
        'selectors = environ.get("NCCL_IB_HCA", "").split(",")',
        "rails=${NCCL_IB_HCA:-}",
        'if [ "$NCCL_IB_HCA" == "x" ]; then',
        "# export NCCL_IB_HCA=rocep1s0f0",
        'echo "fabric $NCCL_IB_HCA"',
    ],
)
def test_inheriting_forwarding_and_reading_pass(code: str) -> None:
    assert guard.scan(code + "\n", {1}) == []


def test_only_added_lines_of_adapter_code_are_judged(tmp_path: Path) -> None:
    (tmp_path / "adapters/a").mkdir(parents=True)
    (tmp_path / "adapters/a/run.sh").write_text(
        "export NCCL_IB_HCA=old\nexport NCCL_IB_HCA=new\n"
    )
    (tmp_path / "adapters/a/NOTES.md").write_text("NCCL_IB_HCA=rocep1s0f0\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/other").write_text("NCCL_IB_HCA=x\n")
    diff = (
        "+++ b/adapters/a/run.sh\n@@ -1,0 +2 @@\n+export NCCL_IB_HCA=new\n"
        "+++ b/adapters/a/NOTES.md\n@@ -0,0 +1 @@\n+NCCL_IB_HCA=rocep1s0f0\n"
        "+++ b/tools/other\n@@ -0,0 +1 @@\n+NCCL_IB_HCA=x\n"
    )
    problems = guard.check(tmp_path, diff)
    assert [p.split(":")[:2] for p in problems] == [["adapters/a/run.sh", "2"]]
