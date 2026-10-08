"""Ended intents release request identities; uncertain reads never duplicate effects."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.lifecycle import LifecycleState
from spark_sweep.prefetch import PIN_RETRY_MAX_SECONDS, PrefetchConfig, pin_alias
from spark_sweep.state import State


def setup(tmp_path: Path):
    fake = FakeRecipe("fresh")
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [fake],
        [FakeModel("m1")],
        prefetch=PrefetchConfig(pin_profile=14, pin_spark="spark-a"),
    )
    sweep.refresh_catalog()
    return sweep, fleet, clock, sweep.recipes[fake.key]


@pytest.mark.parametrize(
    "ending", [LifecycleState.CANCELLED, LifecycleState.SUPERSEDED]
)
@pytest.mark.parametrize("resume", [False, True])
def test_ended_download_releases_key_then_fresh_selector_completes(
    tmp_path: Path,
    ending: LifecycleState,
    resume: bool,
) -> None:
    sweep, fleet, clock, recipe = setup(tmp_path)
    prefetch = sweep.prefetcher
    prefetch._request(recipe, "model")
    old = copy.deepcopy(sweep.state.downloads[recipe.key])
    fleet.ops[old["operation_id"]]["state"] = ending
    prefetch._poll()
    retired = sweep.state.downloads[recipe.key]
    assert "request_key" not in retired
    assert retired["previous_request_key"] == old["request_key"]
    assert retired["terminal_receipt"]["state"] == ending
    assert not prefetch._active(recipe.key)
    assert not prefetch._can_request(recipe, clock.now())
    assert retired["retry_at"] - clock.now() <= PIN_RETRY_MAX_SECONDS
    if resume:
        sweep.state.save()
        sweep.state = State.load(sweep.state.path, clock.now)
        prefetch.state = sweep.state
    clock.sleep(prefetch.config.retry_cooldown)
    assert prefetch._can_request(recipe, clock.now())
    prefetch.tick(
        sweep.recipes,
        sweep.models,
        [recipe],
        sweep.sizes,
        sweep._present(),
        sweep.boost,
        sweep._cached_models(),
    )
    current = sweep.state.downloads[recipe.key]
    assert current["request_key"] != old["request_key"]
    assert current["operation_id"] != old["operation_id"]
    assert current["retry_of_operation_id"] is None
    assert sweep.state.data["download_history"][recipe.key][-1] == retired
    clock.sleep(fleet.download_seconds + 1)
    prefetch._poll()
    assert current["state"] == LifecycleState.SUCCEEDED
    assert not any(
        "observing original request" in e["message"] for e in sweep.state.data["events"]
    )


def test_unreadable_or_foreign_ending_preserves_identity_with_bounded_reads(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = setup(tmp_path)
    prefetch = sweep.prefetcher
    prefetch._request(recipe, "model")
    old = copy.deepcopy(sweep.state.downloads[recipe.key])
    canonical = fleet._op_doc(fleet.ops[old["operation_id"]])
    for status, document in (
        (2, {"error_type": "control_api", "code": "controller.transport_timeout"}),
        (0, {**canonical, "state": LifecycleState.CANCELLED, "id": "foreign"}),
        (0, {**canonical, "state": LifecycleState.SUPERSEDED, "request_id": "foreign"}),
    ):
        fleet.observations.append((("recipe", "progress"), status, document))
        prefetch._poll()
        assert sweep.state.downloads[recipe.key]["request_key"] == old["request_key"]
        assert not prefetch._can_request(recipe, clock.now())
        calls = len(fleet.calls)
        prefetch._poll()
        assert len(fleet.calls) == calls
        clock.sleep(60)
    assert len([c for _, c in fleet.calls if c[:2] == ("recipe", "download")]) == 1


def test_repeated_endings_back_off_to_a_fixed_ceiling(tmp_path: Path) -> None:
    sweep, fleet, clock, recipe = setup(tmp_path)
    delays = []
    keys = set()
    for _ in range(9):
        sweep.prefetcher._request(recipe, "model")
        record = sweep.state.downloads[recipe.key]
        keys.add(record["request_key"])
        fleet.ops[record["operation_id"]]["state"] = LifecycleState.CANCELLED
        sweep.prefetcher._poll()
        delay = record["retry_at"] - clock.now()
        delays.append(delay)
        clock.sleep(delay)
        assert sweep.prefetcher._can_request(recipe, clock.now())
    assert len(keys) == 9
    assert delays == sorted(delays)
    assert max(delays) == PIN_RETRY_MAX_SECONDS


@pytest.mark.parametrize("missing", [False, True])
def test_pin_observation_recovers_current_assignments_without_gating_downloads(
    tmp_path: Path,
    missing: bool,
) -> None:
    sweep, fleet, clock, recipe = setup(tmp_path)
    prefetch = sweep.prefetcher
    fleet.observations.append(
        (
            ("profile", "export"),
            2,
            {
                "error_type": "control_api",
                "code": "controller.not_found"
                if missing
                else "controller.transport_timeout",
            },
        )
    )

    def reconcile():
        present = set(recipe.model_digests)
        plans = prefetch.plan_queue([recipe], sweep.sizes, present, sweep.boost)
        prefetch._reconcile_pins(
            plans, {recipe.key: recipe}, sweep.models, sweep.sizes, present
        )

    reconcile()
    if not missing:
        assert "controller.transport_timeout" in (prefetch.pin_error or "")
        before = len(fleet.calls)
        reconcile()
        assert len(fleet.calls) == before
        assert prefetch._can_request(recipe, clock.now())
        prefetch._request(recipe, "model")
        clock.sleep(prefetch.config.retry_cooldown)
    fleet.profiles.setdefault(14, {"assignments": [], "revision": 1})["assignments"] = [
        {"assignment_name": pin_alias("obsolete")},
        {"assignment_name": "foreign"},
    ]
    reconcile()
    assert {a["assignment_name"] for a in fleet.profiles[14]["assignments"]} == {
        "foreign",
        pin_alias(recipe.key),
    }
    assert prefetch.pin_error is None
    assert prefetch.pin_failures == 0
    assert (
        bool([c for _, c in fleet.calls if c[:2] == ("profile", "configure")])
        == missing
    )


@pytest.mark.parametrize(
    "ending",
    [LifecycleState.FAILED, LifecycleState.CANCELLED, LifecycleState.SUPERSEDED],
)
@pytest.mark.parametrize("projection", [False, True])
def test_terminal_clear_without_stop_receipt_admits_fresh_clear_from_current_fleet(
    tmp_path: Path,
    ending: LifecycleState,
    projection: bool,
) -> None:
    sweep, fleet, clock, _ = setup(tmp_path)
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
    assert sweep._take_over(clock.now())
    old_key, load = next(iter(sweep.state.data["cleanup_loads"].items()))
    app = fleet.apps[load["app_id"]]
    app["state"] = ending
    document = fleet._app_doc(app)
    if not projection:
        document["progress"] = {}
    fleet.observations.append((("profile", "progress"), 0, document))
    sweep.observe_cleanup()
    assert load["terminal_receipt"]["state"] == ending
    # The fleet can change after the ending. The fresh clear must re-read it.
    fleet.runs[0]["run_id"] = "run-new"
    clock.sleep(300)
    assert sweep._take_over(clock.now())
    new_key = next(reversed(sweep.state.data["cleanup_loads"]))
    assert new_key != old_key
    assert (
        sweep.state.data["cleanup_loads"][new_key]["reviewed_stops"]
        != load["reviewed_stops"]
    )
    assert any(p.run_id == "run-new" for p in sweep.fleet.presences)


def test_terminal_receipt_reconnects_after_lost_acceptance_reply(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = setup(tmp_path)
    prefetch = sweep.prefetcher
    prefetch._request(recipe, "model")
    record = sweep.state.downloads[recipe.key]
    old_key = record["request_key"]
    operation = record.pop("operation_id")
    record["requested_intent"] = record["accepted_intent"]
    record["state"] = LifecycleState.OBSERVING
    fleet.ops[operation]["state"] = LifecycleState.CANCELLED
    sweep.state.save()
    sweep.state = State.load(sweep.state.path, clock.now)
    prefetch.state = sweep.state
    prefetch._poll()
    record = sweep.state.downloads[recipe.key]
    assert record["terminal_receipt"]["id"] == operation
    assert record["previous_request_key"] == old_key
    clock.sleep(prefetch.config.retry_cooldown)
    assert prefetch._can_request(recipe, clock.now())
    prefetch._request(recipe, "model")
    assert sweep.state.downloads[recipe.key]["request_key"] != old_key
