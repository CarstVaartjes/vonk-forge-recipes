"""Resume after a crash, Ctrl-C, owner safety, and fixes feeding back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _recipes() -> list[FakeRecipe]:
    return [
        FakeRecipe("a", ("m1",)),
        FakeRecipe("b", ("m2",)),
        FakeRecipe("c", ("m3",)),
    ]


def _models() -> list[FakeModel]:
    return [FakeModel("m1"), FakeModel("m2"), FakeModel("m3")]


def _crash_after(clock, seconds: float, exception: BaseException) -> None:
    start = clock.now()
    fired = []

    def hook(now: float) -> None:
        if now - start >= seconds and not fired:
            fired.append(True)
            raise exception

    clock.hooks.append(hook)


def test_resume_after_a_crash_skips_finished_recipes_and_adopts_the_load(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(tmp_path, _recipes(), _models(), gateway=gateway)

    def reboot(
        _now: float,
    ) -> None:  # the machine dies while the third recipe is loading
        if sweep.state.slots and any(
            e["status"] == "passed" for e in sweep.state.recipes.values()
        ):
            raise RuntimeError("machine rebooted")

    clock.hooks.append(reboot)
    with pytest.raises(RuntimeError):
        sweep.run()
    saved = json.loads((tmp_path / "state.json").read_text())
    passed_before = {k for k, e in saved["recipes"].items() if e["status"] == "passed"}
    assert (
        passed_before and saved["slots"]
    )  # work was finished, and a lane was mid-flight
    clock.hooks.clear()
    again, _, _ = make_sweep(tmp_path, _recipes(), _models(), fleet=fleet)
    assert again.run() == 0
    assert {e["status"] for e in again.state.recipes.values()} == {"passed"}
    # Each recipe was tested exactly once: a recipe qualified before the crash is never added again.
    adds = [c[2] for p, c in fleet.calls if p == 10 and c[:2] == ("profile", "add")]
    assert sorted(adds) == sorted({f"vonk-forge/{s}" for s in "abc"}) and len(adds) == 3
    lines = [
        json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert sorted(x["recipe"] for x in lines) == sorted(
        {f"vonk-forge/{s}" for s in "abc"}
    )


def test_ctrl_c_cancels_the_lane_load_and_requeues(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", load_seconds=900)],
        [FakeModel("m1")],
        gateway=gateway,
    )
    _crash_after(clock, 200, KeyboardInterrupt())
    assert sweep.run() == 130
    cancels = [c for p, c in fleet.calls if c[:2] == ("profile", "cancel")]
    assert (
        cancels
        and cancels[0][3:] == ("--yes", "--detach")
        and {p for p, c in fleet.calls if c[:2] == ("profile", "cancel")} == {10}
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["recipes"]["vonk-forge/a"]["status"] == "pending"  # not blamed
    assert state["slots"] == {} and state["load"] is None
    assert fleet.profiles[10]["assignments"] == []


def test_owner_load_pauses_the_sweep_and_requeues_without_blame(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, _recipes(), _models(), gateway=gateway, owner_hold_seconds=100
    )
    start = clock.now()
    window: list[float] = []

    def owner(now: float) -> None:
        if now - start >= 100 and not window:
            fleet.owner_load(2, duration=1000)
            window.append(now)

    clock.hooks.append(owner)
    assert sweep.run() == 0
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert all(
        e["attempts"] == 1 for e in sweep.state.recipes.values()
    )  # preemption cost no attempt
    # No load was submitted while the owner's application ran (and the hold after it).
    submitted = [
        t
        for t, (p, c) in zip(fleet.call_times, fleet.calls)
        if p == 10 and c[:3] == ("profile", "load", "--yes")
    ]
    assert not [t for t in submitted if window[0] < t < window[0] + 1000 + 100]
    assert not fleet.writes(2)  # nothing the sweep did touched profile 2


def test_restore_owner_profile_loads_it_last(tmp_path: Path, gateway: Gateway) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway, restore_owner=2
    )
    sweep.run()
    last = [(p, c) for p, c in fleet.calls if c[:2] == ("profile", "load")][-1]
    assert last[0] == 2 and "--yes" in last[1]
    assert not [
        c for p, c in fleet.calls if p == 10 and c[:3] == ("profile", "load", "--yes")
    ][1:]  # no stop load: restoring replaces it


def test_a_changed_recipe_document_requeues_only_that_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("good"),
        FakeRecipe("bad", ("m2",), fail_load="driver mismatch", fail_phase="start"),
        FakeRecipe("same", ("m3",)),
    ]
    models = [FakeModel("m1"), FakeModel("m2"), FakeModel("m3")]
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    sweep.run()
    assert sweep.state.recipes["vonk-forge/bad"]["status"] == "failed"
    # A fix lands: the refresh publishes new revisions of `bad` (fixed) and `good` (touched).
    fleet.recipes["vonk-forge/bad"].content = "c2"
    fleet.recipes["vonk-forge/bad"].fail_load = None
    fleet.recipes["vonk-forge/good"].content = "c2"
    again, _, _ = make_sweep(tmp_path, recipes, models, fleet=fleet)
    again.refresh_catalog()
    states = {
        k.split("/")[1]: (e["status"], e.get("requeued_because"))
        for k, e in again.state.recipes.items()
    }
    assert states["bad"] == ("pending", "failed-on-older-revision")
    assert states["good"] == ("pending", "passed-on-older-revision")
    assert states["same"] == ("passed", None)
    # Failures first, revalidation of old passes last.
    again.pending()
    assert again.run() == 0
    assert again.state.recipes["vonk-forge/bad"]["status"] == "passed"
    assert again.state.recipes["vonk-forge/bad"]["content_sha256"] == "c2"
    assert (
        again.state.recipes["vonk-forge/same"]["content_sha256"] == "c1"
    )  # untouched, not retested


def test_no_revalidate_keeps_old_passes(tmp_path: Path, gateway: Gateway) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("good")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    fleet.recipes["vonk-forge/good"].content = "c2"
    again, _, _ = make_sweep(
        tmp_path,
        [FakeRecipe("good")],
        [FakeModel("m1")],
        fleet=fleet,
        revalidate_passes=False,
    )
    again.refresh_catalog()
    assert again.state.recipes["vonk-forge/good"]["status"] == "passed"


def test_the_status_files_describe_lanes_queue_throughput_and_clusters(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("a", ("m1",)),
        FakeRecipe("b", ("m2",), fail_load="driver mismatch", fail_phase="start"),
    ]
    sweep, _, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1"), FakeModel("m2")],
        gateway=gateway,
        status_seconds=0,
    )
    seen: list[dict] = []

    def watch(_now: float) -> None:
        if (tmp_path / "status.json").exists():
            seen.append(json.loads((tmp_path / "status.json").read_text()))

    clock.hooks.append(watch)
    sweep.run()
    midway = next(s for s in seen if any(lane for lane in s["lanes"].values()))
    assert set(midway["lanes"]) == {"spark-a", "spark-b"}
    assert "downloads" in midway and "eta" in midway and "queue" in midway
    final = json.loads((tmp_path / "status.json").read_text())
    assert final["counts"] == {"passed": 1, "failed": 1}
    assert final["failure_clusters"][0]["count"] == 1
    assert "# Hardware sweep status" in (tmp_path / "status.md").read_text()
    assert (
        "| vonk-forge/b | failed | start / start |"
        in (tmp_path / "report.md").read_text()
    )


def test_a_profile_that_already_holds_someone_elses_assignments_is_never_adopted(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.profiles[10] = {
        "revision": 4,
        "assignments": [
            {
                "recipe_selector": "x/y",
                "spark_ids": ["spark-a"],
                "assignment_name": "mine",
                "desired_state": "running",
            }
        ],
        "latest": None,
    }
    with pytest.raises(RuntimeError, match="profile 10 already holds"):
        sweep.run()
    assert not fleet.writes(10)  # not one write to the foreign profile


def test_the_sweep_profiles_are_labelled_so_a_rerun_recognises_them(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    assert fleet.profiles[10]["labels"] == {"purpose": "hardware-sweep"}
    assert fleet.profiles[13]["labels"] == {"purpose": "hardware-sweep"}
