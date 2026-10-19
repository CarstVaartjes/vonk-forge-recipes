"""Saved-state fault injection: an item's wait cannot reserve the other lane."""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

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
