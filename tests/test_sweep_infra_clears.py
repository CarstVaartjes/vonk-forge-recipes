"""An infrastructure entry lasts exactly as long as that work keeps failing.

On hardware the status listed ``owner-wait`` (a source retired in #1350) and
two ENOSPC entries from a disk that had been freed a week earlier, each "next
try in 0m00s". Any entry also counts as a retry in progress (``_nothing_moves``),
which silences the "pending work has no runnable lane" alert.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

SWEEP = Path(__file__).resolve().parents[1] / "spark_sweep"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def test_a_previous_runs_entries_are_gone_once_the_sweep_runs(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")], gateway=gateway
    )
    sweep.state.data["infra"]["owner-wait"] = {
        "count": 1,
        "message": "still paused: profile 1 ownership is unreadable",
        "since": clock.now() - 600_000,
        "until": clock.now() - 599_970,
    }
    assert sweep.run() == 0
    assert sweep.state.data["infra"] == {}
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}


def test_a_failed_save_is_cleared_by_the_next_save(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")], gateway=gateway
    )
    save = sweep.state.save

    def full_disk() -> None:
        raise OSError(28, "No space left on device")

    sweep.state.save = full_disk
    sweep._save_quietly()
    assert "state-save" in sweep.state.data["infra"]
    sweep.state.save = save
    sweep._save_quietly()
    assert "state-save" not in sweep.state.data["infra"]


def _sources(call: str) -> set[str]:
    found = set()
    for path in SWEEP.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == call
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                found.add(str(node.args[0].value))
    return found


def test_every_infrastructure_source_is_cleared_when_its_work_succeeds() -> None:
    # The guard: a source noted on failure and never cleared lists a resolved
    # problem forever and silences the stuck-work alert.
    assert _sources("note_infra") - _sources("clear_infra") == set()


def test_review_recovery_follows_all_admission_and_adoption_checks() -> None:
    tree = ast.parse((SWEEP / "run.py").read_text(encoding="utf-8"))
    apply = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_apply"
    )
    checks = {"_cleanup_adoption", "_adopted_requests", "_blocked"}
    checked_lines = {
        node.func.attr: node.lineno
        for node in ast.walk(apply)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in checks
    }
    assert checked_lines.keys() == checks
    recoveries = [
        node.lineno
        for node in ast.walk(apply)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clear_infra"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "review"
    ]
    assert len(recoveries) == 1
    assert recoveries[0] > max(checked_lines.values())
