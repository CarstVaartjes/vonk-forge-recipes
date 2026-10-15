"""A crash of the sweep's own smoke code is a tool fault: evidence kept, no attempt spent.

The live sweep lost a recipe to ``smoke.crash: FileNotFoundError(2, ...)``: its source
tree lost ``qualification/fixtures`` after it started, the diffusiongemma V_RED case read
its image fixture lazily, and the row kept only ``repr(error)`` and spent the attempt.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import run, verdict
from spark_sweep.definitions import Definitions

DIFFUSIONGEMMA = "vonk-forge/diffusiongemma-26b-nvfp4-r0b0tlab-vllm-single"
QUALIFICATION = Path(__file__).resolve().parents[1] / "qualification"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    if not (tmp_path / "results.jsonl").exists():
        return []
    return [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]


def _read_missing_fixture(*_: Any) -> Any:
    return (Path("/nonexistent-sweep-src") / "fixtures" / "tiny.png.b64").read_bytes()


def test_a_crashing_smoke_keeps_its_evidence_and_spends_no_attempt(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = FakeRecipe("r")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    real_smoke = run.smoke_service
    monkeypatch.setattr(run, "smoke_service", _read_missing_fixture)

    def crashed() -> list[dict[str, Any]]:
        return [r for r in _rows(tmp_path) if r.get("code") == "smoke.crash"]

    for _ in range(2000):
        sweep.tick()
        if crashed():
            break
        clock.sleep(sweep.cfg.poll_seconds)
    (row,) = crashed()

    assert (row["status"], row["disposition"]) == ("requeued", "observe")
    assert row["failure_class"] == verdict.SWEEP_CRASH
    evidence = row["evidence"]
    assert evidence["type"] == "FileNotFoundError"
    assert evidence["filename"] == "/nonexistent-sweep-src/fixtures/tiny.png.b64"
    assert "tiny.png.b64" in evidence["message"]
    assert 1 <= len(evidence["frames"]) <= run.CRASH_FRAMES
    # The worker thread's frames survive ``future.result()``: where it raised is named.
    assert any("_read_missing_fixture" in frame for frame in evidence["frames"])
    assert sweep.state.status(recipe.key) == "pending"
    assert sweep.state.entry(recipe.key).get("attempts", 0) == 0
    assert any(
        "crashed in the sweep" in e["message"] and "tiny.png.b64" in e["message"]
        for e in sweep.state.data["events"]
    )
    assert "smoke-tool" in sweep.state.data["infra"]

    # The tool is fixed: the recipe gets its run, and the tool fault clears on a pass.
    monkeypatch.setattr(run, "smoke_service", real_smoke)
    for _ in range(4000):
        sweep.tick()
        if sweep.state.status(recipe.key) == "passed":
            break
        clock.sleep(sweep.cfg.poll_seconds)
    assert sweep.state.status(recipe.key) == "passed"
    assert "smoke-tool" not in sweep.state.data["infra"]


def test_diffusiongemma_smoke_renders_its_image_fixture() -> None:
    definitions = Definitions.load()
    alias = "diffusiongemma-26b-a4b-it-nvfp4"
    cases = {c["id"]: c for c in definitions.service_cases(DIFFUSIONGEMMA, alias)}
    assert set(cases) == {"M0", "A391", "T_REPORT", "V_RED"}
    content = cases["V_RED"]["body"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_definitions_hold_their_fixtures_when_the_tree_changes(
    tmp_path: Path,
) -> None:
    """A sweep runs for days; its source tree losing fixtures must not crash a smoke."""
    root = tmp_path / "qualification"
    shutil.copytree(QUALIFICATION, root)
    definitions = Definitions.load(root)
    shutil.rmtree(root / "fixtures")
    cases = definitions.service_cases(DIFFUSIONGEMMA, "alias")
    assert any(c["id"] == "V_RED" for c in cases)
