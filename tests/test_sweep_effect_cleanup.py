"""Exact child receipts free disjoint lanes; root timers never replace intent."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import CleanupEnd, CleanupEndReason

from spark_sweep import cleanup, policy
from spark_sweep.lifecycle import LifecycleState

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


def run(alias: str, nodes: list[str]) -> dict:
    return {
        "alias": alias,
        "recipe": "external",
        "run_id": f"run-{alias}",
        "node_ids": nodes,
        "ready": True,
    }


def scenario(tmp_path: Path, *, dual: bool = False):
    recipe = FakeRecipe("ready", local="cached", load_seconds=10)
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    if dual:
        fleet.runs.append(run("gang", ["spk_a", "spk_b"]))
        fleet.cleanup_stop_seconds["gang"] = 3600
    else:
        fleet.runs.extend([run("healthy", ["spk_a"]), run("offline", ["spk_b"])])
        # Recover an older whole-fleet clearing accepted while both nodes were online.
        fleet.cleanup_stop_seconds["offline"] = 3600
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    key, original = next(iter(sweep.state.data["cleanup_loads"].items()))
    app = fleet.apps[original["app_id"]]
    if not dual:
        fleet.offline_nodes.add("spk_b")
        sweep._take_over(clock.now())
    return sweep, fleet, clock, recipe, key, original, app


def mutations(fleet) -> list[tuple]:
    return [
        command
        for profile, command in fleet.calls
        if profile == 10
        and command[:2] == ("profile", "load")
        and "--review" not in command
    ]


def apply_healthy(sweep, clock, recipe) -> None:
    sweep._apply([policy.Placement(sweep.recipes[recipe.key], ("spk_a",))], clock.now())


def test_healthy_lane_does_not_require_offline_cleanup_adoption(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, _key, original, app = scenario(tmp_path)
    assert sweep._cleanup_nodes() == set()
    stopped, pending = original["effects"]
    assert cleanup.stopped(stopped)
    assert pending["state"] == "pending" and pending["target_id"] == "run-offline"
    apply_healthy(sweep, clock, recipe)
    slot = sweep.state.slots[recipe.key]
    current = sweep.state.data["loads"][slot["request_key"]]
    assert current["adopted_cleanup_requests"] == []
    links = fleet.reviewed_cleanup_effects[-1]
    assert any(cleanup.adopted({"effects": {"adopted": links}}, pending) for _ in [0])
    assert app["state"] == "running"
    assert fleet.apps[current["app_id"]]["state"] == "running"
    assert any(
        row["application_id"] == app["id"]
        and row["operation_id"] == pending["operation_id"]
        for row in fleet.apps[current["app_id"]]["effects"]
    )
    assert any(
        command[:2] == ("profile", "load") and "--review" in command
        for _, command in fleet.calls
    )


def test_missing_offline_cleanup_adoption_admits_fresh_load_without_blaming_recipe(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, _key, _original, _app = scenario(tmp_path)
    before = len(mutations(fleet))
    fleet.cleanup_adoption = False
    apply_healthy(sweep, clock, recipe)
    assert len(mutations(fleet)) == before + 1
    assert recipe.key in sweep.state.slots
    assert sweep.state.status(recipe.key) != "failed"
    assert sweep._cleanup_nodes() == set()


def test_fresh_capacity_review_can_refuse_a_locally_cached_free_lane(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, _key, original, _app = scenario(tmp_path)
    before = len(mutations(fleet))
    # Cached local placement facts are stale; the actual new Controller review owns admission.
    fleet.recipes[recipe.key].memory = 200 * 2**30
    apply_healthy(sweep, clock, recipe)
    assert len(mutations(fleet)) == before
    assert recipe.key not in sweep.state.slots
    assert original["effects"][1]["state"] == "pending"
    assert fleet.reviewed_cleanup_effects[-1]


def test_offline_cleanup_does_not_hold_finished_lane_observer(
    tmp_path: Path,
) -> None:
    sweep, _fleet, clock, recipe, _key, _original, app = scenario(tmp_path)
    apply_healthy(sweep, clock, recipe)
    slot = sweep.state.slots[recipe.key]
    request = slot["request_key"]
    current = sweep.state.data["loads"][request]
    slot["phase"] = "finished"
    sweep._release_orphan_load(current)
    assert request not in sweep.state.data["loads"]
    assert current["adopted_cleanup_requests"] == []
    assert app["state"] == "running"
    apply_healthy(sweep, clock, recipe)
    assert recipe.key in sweep.state.slots


def test_pending_root_keeps_exact_child_identity_beyond_old_wait_budget_and_restart(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, key, original, app = scenario(tmp_path)
    previous = copy.deepcopy(original["effects"])
    before = len(mutations(fleet))
    for _ in range(4):
        clock.sleep(300)
        sweep.observe_cleanup()
        assert sweep._take_over(clock.now())
    assert len(mutations(fleet)) == before
    assert original["effects"] == previous
    assert original["request_key"] == key and original["app_id"] == app["id"]
    assert not any(command[:2] == ("profile", "cancel") for _, command in fleet.calls)
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], fleet=fleet
    )
    restarted.preflight()
    restarted.observe_cleanup()
    assert restarted._take_over(clock.now())
    assert restarted.state.data["cleanup_loads"][key]["effects"] == previous
    assert len(mutations(fleet)) == before


@pytest.mark.parametrize(
    "corruption",
    ["missing-projection", "run-mismatch", "scope-mismatch", "receipt-mismatch"],
)
def test_unknown_or_mismatched_full_gang_stop_never_frees_one_member(
    tmp_path: Path, corruption: str
) -> None:
    sweep, fleet, _clock, _recipe, _key, original, app = scenario(tmp_path, dual=True)
    row = app["effects"][0]
    if corruption == "missing-projection":
        fleet.cleanup_projection = False
    elif corruption == "run-mismatch":
        row["target_id"] = "different-run"
    elif corruption == "scope-mismatch":
        row["stop_effect"]["node_ids"] = ["spk_a"]
    else:
        row["state"] = "succeeded"
        row["result"] = {
            "run_switch_operation_id": row["operation_id"],
            "run_switch": {
                "phase_results": [{"phase": "stop", "run_id": "different-run"}]
            },
        }
    sweep.observe_cleanup()
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    assert not cleanup.stopped(original["effects"][0])


def test_gang_claim_occupies_full_topology_until_same_exact_stop_receipt(
    tmp_path: Path,
) -> None:
    sweep, _fleet, clock, _recipe, _key, original, _app = scenario(tmp_path, dual=True)
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    clock.sleep(1800)
    sweep.observe_cleanup()
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    clock.sleep(1800)
    sweep.observe_cleanup()
    assert cleanup.stopped(original["effects"][0])
    assert sweep._cleanup_nodes() == set()


def test_terminal_cleanup_reenters_fresh_admission_from_current_topology(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, _recipe, _key, original, app = scenario(tmp_path, dual=True)
    app["state"] = "cancelled"
    app["effects"][0]["state"] = "cancelled"
    sweep.observe_cleanup()
    fleet.runs[0]["node_ids"] = ["spk_a"]
    fleet.cleanup_stop_seconds["gang"] = 0
    before = len(mutations(fleet))
    assert sweep._take_over(clock.now())
    assert len(mutations(fleet)) == before + 1
    assert mutations(fleet)[-1][-1] != mutations(fleet)[0][-1]
    assert original["effects"][0]["state"] == "cancelled"
    latest = next(reversed(sweep.state.data["cleanup_loads"].values()))
    assert latest["effects"][0]["node_ids"] == ["spk_a"]
    assert cleanup.stopped(latest["effects"][0])
    assert sweep._cleanup_nodes() == set()


def test_older_successful_stop_receipt_cannot_release_newer_stop_intent(
    tmp_path: Path,
) -> None:
    sweep, _fleet, _clock, _recipe, _key, original, _app = scenario(tmp_path)
    older = original["effects"][0]
    assert cleanup.stopped(older)
    newer = copy.deepcopy(older)
    newer.update(
        effect_id="new-owner:queue:0:stop:run-healthy",
        application_id="new-owner",
        workload_intent_ordinal=older["workload_intent_ordinal"] + 1,
        request_key="new-request",
        operation_id="new-operation",
        state="pending",
        result=None,
    )
    sweep.state.data["cleanup_loads"]["new-request"] = {
        "request_key": "new-request",
        "app_id": "new-owner",
        "projection_known": True,
        "state": "running",
        "effects": [newer],
    }
    assert sweep._cleanup_nodes() == {"spk_a"}


def test_missing_cleanup_binding_keeps_one_incident_across_restart(
    tmp_path: Path,
) -> None:
    from spark_sweep.report import build_status, render_status

    sweep, fleet, clock, recipe, key, original, app = scenario(tmp_path, dual=True)
    fleet.cleanup_adoption = False
    before = len(mutations(fleet))
    first_seen = clock.now()
    counts = []
    delays = []
    for cycle in range(3):
        apply_healthy(sweep, clock, recipe)
        incident = sweep.state.data["infra"]["review"]
        counts.append(incident["count"])
        delays.append(incident["until"] - clock.now())
        assert incident["since"] == first_seen
        assert len(mutations(fleet)) == before
        assert recipe.key not in sweep.state.slots
        assert not any(
            "problem cleared (review)" in event["message"]
            for event in sweep.state.data["events"]
        )
        if cycle == 1:
            sweep.state.save()
            sweep, _, _ = make_sweep(
                tmp_path, [recipe], [FakeModel("m1", local="cached")], fleet=fleet
            )

            class RestartObserved(BaseException):
                pass

            def stop_after_restart(restarted=sweep, saved=incident) -> None:
                assert restarted.state.data["infra"]["review"] == saved
                raise RestartObserved

            sweep.check_client = stop_after_restart
            with pytest.raises(RestartObserved):
                sweep.run()
            sweep.preflight()
            sweep.refresh_catalog()
        clock.sleep(delays[-1])
    assert counts == [1, 2, 3]
    assert delays == [30, 60, 120]
    incident = sweep.state.data["infra"]["review"]
    assert incident["escalate_at"] == first_seen + sweep.cfg.idle_alert_seconds
    status = build_status(sweep)
    page = render_status(status)
    assert "escalat" in page.lower()
    effect = original["effects"][0]
    for identity in (key, app["id"], effect["effect_id"], effect["operation_id"]):
        assert identity in page
    clock.sleep(incident["escalate_at"] - clock.now())
    status = build_status(sweep)
    assert any(
        "ESCALATION" in alert and app["id"] in alert for alert in status["alerts"]
    )
    fleet.cleanup_adoption = True
    apply_healthy(sweep, clock, recipe)
    assert recipe.key in sweep.state.slots
    assert len(mutations(fleet)) == before + 1
    assert "review" not in sweep.state.data["infra"]
    assert (
        sum(
            "problem cleared (review)" in event["message"]
            for event in sweep.state.data["events"]
        )
        == 1
    )
    assert not build_status(sweep)["alerts"]


@pytest.mark.parametrize("with_dependency", [False, True])
def test_polling_another_load_does_not_clear_review_wait(
    tmp_path: Path, with_dependency: bool
) -> None:
    sweep, _fleet, clock, recipe, key, original, _app = scenario(tmp_path)
    apply_healthy(sweep, clock, recipe)
    dependency = (
        f"cleanup request={key}, operation={original['effects'][1]['operation_id']}"
        if with_dependency
        else None
    )
    sweep.note_infra("review", "waiting for cleanup binding", dependency=dependency)
    incident = copy.deepcopy(sweep.state.data["infra"]["review"])
    sweep.poll_application()
    assert sweep.state.data["infra"]["review"] == incident
    assert not any(
        "problem cleared (review)" in event["message"]
        for event in sweep.state.data["events"]
    )


@pytest.mark.parametrize(
    "owner_state", [LifecycleState.FAILED, LifecycleState.SUPERSEDED]
)
@pytest.mark.parametrize("run_present", [False, True])
@pytest.mark.parametrize("terminal_receipt", [False, True])
def test_ended_cleanup_owner_releases_review_or_gets_a_new_exact_stop(
    tmp_path: Path,
    owner_state: LifecycleState,
    run_present: bool,
    terminal_receipt: bool,
) -> None:
    sweep, fleet, clock, recipe, key, original, app = scenario(tmp_path, dual=True)
    original["effects"][0]["operation_id"] = app["effects"][0]["operation_id"] = None
    fleet.cleanup_adoption = False
    apply_healthy(sweep, clock, recipe)
    assert "review" in sweep.state.data["infra"]
    app["state"] = owner_state
    app["effects"][0].update(state=LifecycleState.CANCELLED, operation_id=None)
    sweep.observe_cleanup()  # the parent ended without stopping its exact run
    assert original["terminal_receipt"]["state"] == owner_state
    if not terminal_receipt:
        # Persisted state can predate receipt retention.
        original.pop("terminal_receipt")
    before = len(mutations(fleet))
    if not run_present:
        fleet.runs.clear()  # another Controller decision already achieved the stop
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], fleet=fleet
    )
    restarted.preflight()
    restarted.refresh_catalog()
    # Exercise the review that was blocked before restart, without waiting for
    # another cleanup observation to retire its owner first.
    apply_healthy(restarted, clock, recipe)
    assert recipe.key in restarted.state.slots
    assert len(mutations(fleet)) == before + 1
    assert "review" not in restarted.state.data["infra"]
    assert (
        sum(
            "problem cleared (review)" in event["message"]
            for event in restarted.state.data["events"]
        )
        == 1
    )
    record = restarted.state.data["cleanup_loads"][key]
    assert record["effects"][0]["state"] == LifecycleState.CANCELLED
    current = restarted.state.data["loads"][
        restarted.state.slots[recipe.key]["request_key"]
    ]
    assert current["adopted_cleanup_requests"] == []
    if run_present:
        assert "end" not in record
        own_stop = fleet.apps[current["app_id"]]["effects"][0]
        assert own_stop["application_id"] == current["app_id"] != app["id"]
        assert own_stop["operation_id"]
        assert cleanup.stop_signature(
            own_stop["stop_effect"]
        ) == cleanup.stop_signature(original["effects"][0]["stop_effect"])
    else:
        assert (
            CleanupEnd.model_validate(record["end"]).reason
            is CleanupEndReason.TARGET_ABSENT
        )
        retired = [
            e for e in restarted.state.data["events"] if "retired:" in e["message"]
        ]
        assert len(retired) == 1
        restarted.observe_cleanup()
        assert [
            e for e in restarted.state.data["events"] if "retired:" in e["message"]
        ] == retired


@pytest.mark.parametrize(
    "observation", ["fresh", "different-run", "offline", "incomplete", "unavailable"]
)
def test_cleanup_review_uses_fresh_exact_run_absence(
    tmp_path: Path, observation: str
) -> None:
    sweep, fleet, clock, recipe, _key, original, _app = scenario(tmp_path, dual=True)
    fleet.cleanup_adoption = False
    apply_healthy(sweep, clock, recipe)
    fleet.runs.clear()  # the cached sweep.fleet still shows the old run
    if observation == "different-run":
        replacement = run("gang", ["spk_a", "spk_b"])
        replacement["run_id"] = "replacement-run"
        fleet.runs.append(replacement)
    elif observation == "offline":
        fleet.offline_nodes.add("spk_b")
    elif observation == "incomplete":
        fleet.observations.append((("fleet",), 0, {"nodes": []}))
    elif observation == "unavailable":
        fleet.observations.append((("fleet",), 124, {}))
    before = len(mutations(fleet))
    apply_healthy(sweep, clock, recipe)
    recovered = observation in ("fresh", "different-run")
    assert (recipe.key in sweep.state.slots) is recovered
    assert ("end" in original) is recovered
    assert len(mutations(fleet)) == before + int(recovered)
    assert ("review" not in sweep.state.data["infra"]) is recovered
    assert not cleanup.stopped(original["effects"][0])


def saved_confirmed_cleanup(
    tmp_path: Path,
    saved_state: LifecycleState,
    owner_state: LifecycleState,
    empty: bool,
):
    sweep, fleet, clock, recipe, key, original, app = scenario(tmp_path, dual=True)
    clock.sleep(3600)
    reply = sweep.vk.run("profile", "progress", "--application", app["id"], profile=10)
    rows = cleanup.stop_effects(reply.document)
    assert rows and all(cleanup.stopped(row) for row in rows)
    original["effects"] = rows
    if empty:
        original.update(effects=[], reviewed_stops=[])
        app["effects"] = []
    original.update(state=saved_state, projection_known=True)
    original.pop("terminal_receipt", None)
    app["state"] = owner_state
    # Hold the parent's projection active after all its children finished.
    app["blocked"] = owner_state in (LifecycleState.RUNNING, LifecycleState.QUEUED)
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], fleet=fleet
    )
    restarted.preflight()
    restarted.refresh_catalog()
    fleet.calls.clear()
    return restarted, fleet, clock, recipe, key, app


@pytest.mark.parametrize("saved_state", [LifecycleState.RUNNING, LifecycleState.QUEUED])
@pytest.mark.parametrize(
    "owner_state",
    [
        LifecycleState.SUCCEEDED,
        LifecycleState.FAILED,
        LifecycleState.RUNNING,
        LifecycleState.QUEUED,
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_saved_confirmed_cleanup_reobserves_owner_and_retires(
    tmp_path: Path,
    saved_state: LifecycleState,
    owner_state: LifecycleState,
    empty: bool,
) -> None:
    sweep, fleet, clock, recipe, key, app = saved_confirmed_cleanup(
        tmp_path, saved_state, owner_state, empty
    )
    sweep.note_infra("clearing", "stale cleanup owner")
    sweep.observe_cleanup()
    record = sweep.state.data["cleanup_loads"][key]
    assert record["state"] == owner_state
    assert (
        CleanupEnd.model_validate(record["end"]).reason
        is CleanupEndReason.TARGET_ABSENT
    )
    reads = [c for _, c in fleet.calls if c[:2] == ("profile", "progress")]
    assert len(reads) == 1 and app["id"] in reads[0]
    assert not mutations(fleet)
    assert not any(c[:2] == ("profile", "cancel") for _, c in fleet.calls)
    calls = len(fleet.calls)
    sweep.observe_cleanup()
    assert len(fleet.calls) == calls  # retired history does not generate I/O
    assert "clearing" not in sweep.state.data["infra"]
    assert sweep.backoff_until <= clock.now()
    assert sweep._take_over(clock.now())
    apply_healthy(sweep, clock, recipe)
    assert recipe.key in sweep.state.slots
    assert mutations(fleet)


@pytest.mark.parametrize("observation", ["present", "offline", "unavailable"])
def test_confirmed_cleanup_waits_for_fresh_absence_with_bounded_reads(
    tmp_path: Path, observation: str
) -> None:
    sweep, fleet, clock, _recipe, key, _app = saved_confirmed_cleanup(
        tmp_path, LifecycleState.RUNNING, LifecycleState.RUNNING, False
    )
    if observation == "present":
        fleet.runs.append(run("gang", ["spk_a", "spk_b"]))
    elif observation == "offline":
        fleet.offline_nodes.add("spk_a")
    else:
        fleet.observations.append((("fleet",), 0, {}))
    sweep.observe_cleanup()
    record = sweep.state.data["cleanup_loads"][key]
    assert "end" not in record
    budget = record["observation_budget"]
    assert 0 < budget["next_check"] - clock.now() <= 600
    calls = len(fleet.calls)
    sweep.observe_cleanup()
    assert len(fleet.calls) == calls
    fleet.runs.clear()
    fleet.offline_nodes.clear()
    clock.sleep(budget["next_check"] - clock.now())
    sweep.observe_cleanup()
    assert "end" in record


def test_empty_queued_cleanup_has_no_targets_beside_a_foreign_run(
    tmp_path: Path,
) -> None:
    sweep, fleet, _clock, _recipe, key, _app = saved_confirmed_cleanup(
        tmp_path, LifecycleState.QUEUED, LifecycleState.QUEUED, True
    )
    fleet.runs.append(run("foreign", ["spk_a"]))
    sweep.observe_cleanup()
    assert "end" in sweep.state.data["cleanup_loads"][key]
    assert any(row["alias"] == "foreign" for row in fleet.runs)
    assert not mutations(fleet)


def test_catalog_change_preserves_accepted_sibling_lane_and_load(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    recipes = [
        FakeRecipe("changing", local="cached"),
        FakeRecipe("sibling", local="cached"),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    sweep._apply(
        [
            policy.Placement(sweep.recipes[recipes[0].key], ("spk_a",)),
            policy.Placement(sweep.recipes[recipes[1].key], ("spk_b",)),
        ],
        clock.now(),
    )
    sibling = copy.deepcopy(sweep.state.slots[recipes[1].key])
    loads = copy.deepcopy(sweep.state.data["loads"])
    before = list(mutations(fleet))
    sweep.recipes[recipes[0].key] = replace(
        sweep.recipes[recipes[0].key], content_sha256="c2"
    )
    sweep._reconcile_all()
    assert sweep.state.slots[recipes[1].key] == sibling
    assert sweep.state.data["loads"] == loads
    assert mutations(fleet) == before
