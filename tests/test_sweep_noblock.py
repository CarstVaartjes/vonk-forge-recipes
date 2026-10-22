"""Saved-state fault injection: an item's wait cannot reserve the other lane."""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import (
    SweepSlotPhase,
    SweepWait,
    SweepWaitCode,
    SweepWaitReason,
)

from spark_sweep import policy, verdict
from spark_sweep.lifecycle import LifecycleState
from spark_sweep.owner import OwnerStatus
from spark_sweep.report import build_status
from spark_sweep.smoke import SmokeResult
from spark_sweep.vonkctl import VonkctlError

FIXTURE = Path(__file__).parent / "fixtures/sweep-noblock.json"
TRANSPORT_TIMEOUT = json.loads(
    (FIXTURE.parent / "sweep-verdicts/cli-transport-timeout.json").read_text()
)


def test_new_single_lanes_have_independent_application_receipts(tmp_path: Path) -> None:
    """Catches batching two independently placeable lanes into one admission wait."""
    recipes = [FakeRecipe("a", local="cached"), FakeRecipe("b", local="cached")]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._apply(
        [
            policy.Placement(sweep.recipes[recipes[0].key], ("spk_a",)),
            policy.Placement(sweep.recipes[recipes[1].key], ("spk_b",)),
        ],
        clock.now(),
    )
    requests = {slot["request_key"] for slot in sweep.state.slots.values()}
    assert len(requests) == 2
    assert len(sweep.state.slots) == 2
    assert all(
        len([s for s in sweep.state.slots.values() if s["request_key"] == request]) == 1
        for request in requests
    )
    assert not any(args[:2] == ("profile", "cancel") for _, args in fleet.calls)


@pytest.mark.parametrize(
    "admitted,reclaimable", [(False, 0), (False, 300_000_000_000), (True, 0)]
)
def test_saved_unadmitted_storage_wait_keeps_the_same_load(
    tmp_path: Path, admitted: bool, reclaimable: int
) -> None:
    """Pending storage is not a final capacity decision, including after restart."""
    recipe = FakeRecipe("a", node_count=2, local="cached", load_seconds=600)
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._apply(
        [policy.Placement(sweep.recipes[recipe.key], ("spk_a", "spk_b"))],
        clock.now(),
    )
    request = sweep.state.slots[recipe.key]["request_key"]
    application = sweep.state.data["loads"][request]["app_id"]
    document = json.loads((FIXTURE.parent / "sweep-pending-storage.json").read_text())
    document["id"] = application
    document["progress"]["blockers"][0]["storage"]["reclaimable_bytes"] = reclaimable
    if admitted:
        document["progress"]["effects"] = [{"effect_id": "accepted-child"}]
    deadline = sweep.state.slots[recipe.key]["deadline"]
    submissions = sum(
        args[:2] == ("profile", "load") and "--review" not in args
        for _, args in fleet.calls
    )
    for cycle in range(4):
        fleet.observations.append(
            (("profile", "progress", "--application", application), 0, document)
        )
        sweep.tick()
        slot = sweep.state.slots[recipe.key]
        assert slot["phase"] == SweepSlotPhase.LOADING
        assert slot["request_key"] == request
        assert slot["deadline"] == deadline
        wait = SweepWait.model_validate(slot["watch"]["wait"])
        assert wait.reason == SweepWaitReason.ADMISSION
        assert wait.detail == document["progress"]["blockers"][0]["code"]
        if cycle == 1:
            sweep.state.save()
            sweep, _, _ = make_sweep(
                tmp_path, [recipe], list(fleet.models.values()), fleet=fleet
            )
            sweep.refresh_catalog()
        clock.sleep(10)
    # Admission begins the NAS -> Spark copy under the original receipt.
    copying = fleet._app_doc(fleet.apps[application])
    copying["progress"]["child_progress"] = {"phase": "target-copy", "bytes": 100}
    fleet.observations.append(
        (("profile", "progress", "--application", application), 0, copying)
    )
    sweep.tick()
    assert sweep.state.slots[recipe.key]["copying"]
    assert sweep.state.slots[recipe.key]["request_key"] == request
    assert sweep.state.data["loads"][request]["app_id"] == application
    assert (
        sum(
            args[:2] == ("profile", "load") and "--review" not in args
            for _, args in fleet.calls
        )
        == submissions
    )
    assert not list(sweep.log.entries())
    assert not any(args[:2] == ("profile", "cancel") for _, args in fleet.calls)


def test_pending_storage_expires_with_evidence_without_recipe_attempt(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("a", node_count=2, local="cached", load_seconds=10000)
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._apply(
        [policy.Placement(sweep.recipes[recipe.key], ("spk_a", "spk_b"))], clock.now()
    )
    slot = sweep.state.slots[recipe.key]
    request = slot["request_key"]
    application = sweep.state.data["loads"][request]["app_id"]
    document = json.loads((FIXTURE.parent / "sweep-pending-storage.json").read_text())
    document["id"] = application
    for expired in (False, True):
        fleet.observations.append(
            (("profile", "progress", "--application", application), 0, document)
        )
        sweep.advance_slots(clock.now())
        if not expired:
            assert slot["watch"]["wait_deadline"] <= slot["deadline"]
            clock.sleep(slot["deadline"] - clock.now() + 1)
    assert slot["phase"] == SweepSlotPhase.FINISHED
    assert request in sweep.state.data["loads"]
    assert sweep.state.entry(recipe.key).get("attempts", 0) == 0
    row = list(sweep.log.entries())[-1]
    assert row["code"] == SweepWaitCode.BUDGET_EXHAUSTED
    assert row["controller_document"]["id"] == application
    assert not any(args[:2] == ("profile", "cancel") for _, args in fleet.calls)


def test_owner_request_between_lane_submissions_steers_the_next_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipes = [FakeRecipe("a", local="cached"), FakeRecipe("b", local="cached")]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    submit = sweep._submit_load

    def owner_after_first(*args, **kwargs):
        document = submit(*args, **kwargs)
        fleet.owner_load(1, duration=300)
        return document

    monkeypatch.setattr(sweep, "_submit_load", owner_after_first)
    sweep._apply(
        [
            policy.Placement(sweep.recipes[recipes[0].key], ("spk_a",)),
            policy.Placement(sweep.recipes[recipes[1].key], ("spk_b",)),
        ],
        clock.now(),
    )
    assert sweep.owner_status.paused
    assert not any(
        args[:2] == ("profile", "add") and recipes[1].key in args
        for _, args in fleet.calls
    )
    assert not any(args[:2] == ("profile", "cancel") for _, args in fleet.calls)


@pytest.mark.parametrize(
    "source",
    [
        "review",
        "clearing",
        "startup",
        "profile-edit",
        "load-submit",
        "observe",
        "endpoint",
        "smoke-tool",
        "client-key",
        "client-update",
        "download",
        "tick",
        "loop",
        "state-save",
        "results-log",
        "status",
    ],
)
def test_saved_incidents_do_not_back_off_unrelated_cached_work(
    tmp_path: Path, source: str
) -> None:
    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    recipe = FakeRecipe("ready", local="cached")
    sweep, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.note_infra(source, "one operation is unavailable")
    sweep.tick()
    assert recipe.key in sweep.state.slots
    assert not sweep.paused
    assert "paused_at" not in sweep.state.data["owner"]


@pytest.mark.parametrize("alarm", ["application", "platform", "unreadable"])
def test_stuck_receipt_keeps_only_its_lane_after_restart(
    tmp_path: Path, alarm: str
) -> None:
    held = FakeRecipe("held", local="cached", blocked=True, blocked_for=3600)
    ready = FakeRecipe("ready", local="cached")
    gang = FakeRecipe("00-gang", node_count=2, local="cached")
    recipes = [held, ready, gang]
    models = [FakeModel("m1", local="cached")]
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models, boost=(gang.slug,))
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._apply([policy.Placement(sweep.recipes[held.key], ("spk_a",))], clock.now())
    request = sweep.state.slots[held.key]["request_key"]
    application = sweep.state.data["loads"][request]["app_id"]
    if alarm == "application":
        fleet.apps[application]["blockers"] = [
            {"code": verdict.STALLED_CODE, "severity": "error", "detail": "held child"}
        ]
    elif alarm == "platform":
        fleet.stalled_applications = [
            {"application_id": application, "detail": "held child"}
        ]
    else:
        fleet.observations.append(
            (
                ("profile", "progress", "--application", application),
                2,
                TRANSPORT_TIMEOUT,
            )
        )
    sweep.state.data["infra"].update(json.loads(FIXTURE.read_text())["infra"])
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, recipes, models, fleet=fleet, boost=(gang.slug,)
    )
    restarted.tick()
    assert ready.key in restarted.state.slots
    assert restarted.state.slots[ready.key]["node_ids"] == ["spk_b"]
    assert restarted.state.slots[held.key]["request_key"] == request
    assert restarted.state.entry(held.key).get("attempts", 0) == 0
    assert not restarted.paused
    assert not any(args[:2] == ("profile", "cancel") for _, args in fleet.calls)
    if alarm != "unreadable":
        assert application in " ".join(build_status(restarted)["alerts"])


def test_a_failed_profile_edit_replans_the_other_lane_in_the_same_tick(
    tmp_path: Path,
) -> None:
    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    recipes = [FakeRecipe("a", local="cached"), FakeRecipe("b", local="cached")]
    sweep, fleet, _ = make_sweep(tmp_path, recipes, [FakeModel("m1", local="cached")])
    fleet.faults = [(("profile", "add"), TRANSPORT_TIMEOUT["code"], "one edit failed")]
    sweep.tick()
    assert recipes[1].key in sweep.state.slots
    assert recipes[0].key not in sweep.state.slots
    assert sweep.state.entry(recipes[0].key)["not_before"] > sweep.clock.now()


@pytest.mark.parametrize(
    "fault", ["pending", "unreadable", "unknown-projection", "ended"]
)
def test_saved_cleanup_on_one_spark_admits_the_other_in_one_tick(
    tmp_path: Path, fault: str
) -> None:
    recipe = FakeRecipe("ready", local="cached")
    models = [FakeModel("m1", local="cached")]
    sweep, fleet, clock = make_sweep(tmp_path, [recipe], models)
    fleet.runs.append(
        {
            "alias": "external",
            "recipe": "external",
            "run_id": "run-external",
            "node_ids": ["spk_a"],
            "ready": True,
        }
    )
    fleet.cleanup_stop_seconds["external"] = 3600
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    (load,) = sweep.state.data["cleanup_loads"].values()
    application = load["app_id"]
    if fault == "unreadable":
        fleet.observations.append(
            (
                ("profile", "progress", "--application", application),
                2,
                TRANSPORT_TIMEOUT,
            )
        )
    elif fault == "unknown-projection":
        load.pop("projection_known", None)
        fleet.cleanup_projection = False
    elif fault == "ended":
        fleet.apps[application]["state"] = LifecycleState.CANCELLED
        fleet.apps[application]["effects"][0]["state"] = LifecycleState.CANCELLED
        sweep.observe_cleanup()
        fleet.cleanup_adoption = False
    sweep.state.data["infra"].update(json.loads(FIXTURE.read_text())["infra"])
    sweep.state.save()
    restarted, _, _ = make_sweep(tmp_path, [recipe], models, fleet=fleet)
    restarted.tick()
    assert recipe.key in restarted.state.slots
    assert restarted.state.slots[recipe.key]["node_ids"] == ["spk_b"]


def test_fresh_owner_observation_clears_a_global_pause_without_waiting_for_poll(
    tmp_path: Path,
) -> None:
    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    recipe = FakeRecipe("ready", local="cached")
    sweep, fleet, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], owner_poll_seconds=600
    )
    fleet.faults = [
        (("profile", "progress"), TRANSPORT_TIMEOUT["code"], "owner unknown")
    ]
    sweep.tick()
    assert sweep.paused
    assert recipe.key not in sweep.state.slots
    sweep.tick()  # same clock; a fresh success clears the failure
    assert not sweep.paused
    assert recipe.key in sweep.state.slots


def test_an_active_owner_load_stays_global_and_names_its_application(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("ready", local="cached")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    fleet.owner_load(2, duration=300)
    sweep.tick()
    assert sweep.paused
    assert "profile 2" in sweep.pause_reason
    assert not sweep.state.slots


def test_a_stale_global_pause_is_replaced_by_a_fresh_success(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    recipe = FakeRecipe("ready", local="cached")
    sweep, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.owner_status = OwnerStatus(True, "old ownership observation failed")
    sweep.tick()
    assert not sweep.paused
    assert recipe.key in sweep.state.slots
    assert LifecycleState.OBSERVING not in {
        e["status"] for e in sweep.state.recipes.values()
    }


def test_one_smoke_crash_does_not_delay_the_other_lane(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    held = FakeRecipe("held", local="cached")
    ready = FakeRecipe("ready", local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [held, ready], [FakeModel("m1", local="cached")]
    )
    sweep.refresh_catalog()
    sweep._apply([policy.Placement(sweep.recipes[held.key], ("spk_a",))], clock.now())
    sweep.state.slots[held.key]["phase"] = "smoking"
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, [held, ready], [FakeModel("m1", local="cached")], fleet=fleet
    )
    future: Future[SmokeResult] = Future()
    future.set_exception(FileNotFoundError("one smoke fixture disappeared"))
    restarted.futures[held.key] = future
    restarted.tick()
    assert ready.key in restarted.state.slots
    assert restarted.state.entry(held.key)["not_before"] > clock.now()


def test_startup_reobserves_after_one_poll_and_reports_the_exact_wait(
    tmp_path: Path,
) -> None:
    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    sweep, _, clock = make_sweep(tmp_path, [], [])

    def unavailable() -> None:
        raise VonkctlError(("fleet",), None, "fleet is unreadable")

    before = clock.now()
    assert not sweep._startup(unavailable)
    assert clock.now() - before == sweep.cfg.poll_seconds
    page = (tmp_path / "status.md").read_text()
    assert "unavailable" in page and "fleet is unreadable" in page
    assert sweep._startup(lambda: None)
    assert "startup" not in sweep.state.data["infra"]


def test_fleet_outage_blocks_submission_only_until_the_next_fresh_tick(
    tmp_path: Path,
) -> None:
    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    recipe = FakeRecipe("ready", local="cached")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    original = fleet.runner

    def unavailable(argv, timeout):
        if "fleet" in argv:
            return (
                2,
                json.dumps(TRANSPORT_TIMEOUT),
                "",
            )
        return original(argv, timeout)

    sweep.vk.runner = unavailable
    sweep.tick()
    assert not sweep.state.slots
    assert "fleet observation" in sweep.state.data["clearing_reason"]
    sweep.vk.runner = original
    sweep.tick()
    assert recipe.key in sweep.state.slots


def test_storage_shortfall_never_submits_an_impossible_lane(tmp_path: Path) -> None:
    """A waitable batch must not admit a Spark whose free plus evictable disk is short."""
    from vonk_forge_contracts.sweep import SweepStorageCode, SweepStorageWait

    (tmp_path / "state.json").write_text(FIXTURE.read_text())
    recipes = [FakeRecipe("a", local="cached"), FakeRecipe("b", local="cached")]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    blocked_review = json.loads(
        (FIXTURE.parent / "sweep-storage-shortfall.json").read_text()
    )
    fleet.observations.append((("profile", "load", "--review"), 0, blocked_review))
    sweep._apply(
        [
            policy.Placement(sweep.recipes[recipes[0].key], ("spk_a",)),
            policy.Placement(sweep.recipes[recipes[1].key], ("spk_b",)),
        ],
        clock.now(),
    )
    assert recipes[0].key not in sweep.state.slots
    assert recipes[1].key in sweep.state.slots
    wait = SweepStorageWait.model_validate(
        sweep.state.entry(recipes[0].key)["storage_wait"]
    )
    assert wait.reasons[0].code == SweepStorageCode.INSUFFICIENT_DISK
    assert wait.node_ids == {"spk_a"}
    assert sweep.state.entry(recipes[0].key).get("attempts", 0) == 0
    # The saved decision constrains the fallback too, but a new observation
    # after the check interval can reconsider that Spark without manual edits.
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, recipes, list(fleet.models.values()), fleet=fleet
    )
    restarted.refresh_catalog()
    blocked_bin = next(b for b in restarted._bins() if b.id == "spk_a")
    assert recipes[0].key in blocked_bin.disk_blocked_recipes
    clock.sleep(sweep.cfg.defer_delay + 1)
    assert not next(
        b for b in restarted._bins() if b.id == "spk_a"
    ).disk_blocked_recipes
