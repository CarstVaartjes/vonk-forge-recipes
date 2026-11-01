"""Evidence binds to the bytes that run: the review's revision is checked before any load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep import policy
from spark_sweep.qualification import backfill, from_history

OOM = (
    "runtime_process_exited: exit_cause=oom_killed exception=CUDA error: out of memory"
)


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in (path / "results.jsonl").read_text().splitlines()]


def _loads(sweep, calls) -> list:
    return [
        command
        for profile, command in calls
        if profile == sweep.cfg.sweep_profile
        and command[:2] == ("profile", "load")
        and "--yes" in command
    ]


def test_a_review_binding_another_revision_submits_no_load_and_waits_paced(
    tmp_path: Path,
) -> None:
    fake = FakeRecipe("drift", local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [fake], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    fleet.review_revisions[fake.key] = "rev-reverted"
    recipe = sweep.recipes[fake.key]
    before = len(fleet.calls)
    delays = []
    for _ in range(2):
        sweep._apply([policy.Placement(recipe, ("spk_a",))], clock.now())
        delays.append(sweep.state.entry(fake.key)["not_before"] - clock.now())
    assert not _loads(sweep, fleet.calls[before:])
    assert fake.key not in sweep.state.slots
    rows = [r for r in _rows(tmp_path) if r["step"] == "identity"]
    assert [(r["status"], r["blame"], r["code"]) for r in rows] == [
        ("waiting", "platform", policy.REVISION_MISMATCH)
    ] * 2
    assert rows[0]["evidence"]["reviewed_revision"] == "rev-reverted"
    assert delays[1] > delays[0]  # repeats back off instead of reloading every delay
    assert not sweep.qualification_evidence.get(fake.key)


@pytest.mark.parametrize("fails", [False, True])
def test_a_matching_review_runs_and_records_evidence_even_when_the_load_fails(
    tmp_path: Path, gateway: Gateway, fails: bool
) -> None:
    fake = FakeRecipe(
        "fine",
        ("1" * 64,),
        content="c" * 64,
        fail_load=OOM if fails else None,
        fail_phase="start",
    )
    sweep, _, _ = make_sweep(tmp_path, [fake], [FakeModel("1" * 64)], gateway=gateway)
    sweep.run()
    smoke = [r for r in _rows(tmp_path) if r["step"] == "smoke"]
    assert smoke
    assert {r["ran_revision"] for r in smoke} == {f"rev-{fake.content}"}
    assert sweep.qualification_evidence[fake.key].records
    assert "revision_mismatches" not in sweep.state.entry(fake.key)


def test_a_row_whose_run_differs_from_its_identity_is_never_evidence(
    tmp_path: Path, gateway: Gateway
) -> None:
    fake = FakeRecipe("fine", ("1" * 64,), content="c" * 64)
    sweep, _, _ = make_sweep(tmp_path, [fake], [FakeModel("1" * 64)], gateway=gateway)
    sweep.run()
    passed = next(r for r in _rows(tmp_path) if r["step"] == "smoke")
    assert from_history(passed) is not None
    assert from_history({k: v for k, v in passed.items() if k != "ran_revision"})
    wrong = {**passed, "ran_revision": "rev-reverted"}
    assert from_history(wrong) is None
    # The publisher backfill reads the same log rows.
    history = tmp_path / "wrong.jsonl"
    history.write_text(json.dumps(wrong) + "\n")
    result = backfill([history], tmp_path)
    assert (result.read, result.imported) == (1, 0)
    # A restart replays the log through the same reader.
    restart = tmp_path / "restart"
    restart.mkdir()
    (restart / "results.jsonl").write_text(json.dumps(wrong) + "\n")
    again, _, _ = make_sweep(restart, [fake], [FakeModel("1" * 64)])
    assert not again.qualification_evidence.get(fake.key)
