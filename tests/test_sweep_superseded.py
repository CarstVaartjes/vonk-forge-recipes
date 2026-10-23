"""A load the Controller replaced is followed or requeued, never recorded as a recipe failure."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts.sweep import SweepSlotPhase

from spark_sweep import policy
from spark_sweep.identity import AcceptedRequest
from spark_sweep.lifecycle import LifecycleState, RecipeStatus, ReconciliationStatus
from spark_sweep.smoke import SmokeResult
from spark_sweep.vonkctl import Reply

SID = "8cb1d3a7-0000-4000-8000-000000000001"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def test_an_automatic_retry_is_adopted_and_the_recipe_passes(tmp_path, gateway) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.supersede = ["retry"]
    assert sweep.run() == 0
    entry = _entry(sweep, "a")
    assert entry["status"] == "passed"
    assert "following" in str(sweep.state.data["events"])


def test_a_superseded_intent_without_successor_is_requeued_not_failed(
    tmp_path, gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.supersede = ["intent"]
    assert sweep.run() == 0
    assert _entry(sweep, "a")["status"] == "passed"
    assert "load superseded" in str(sweep.state.data["events"])


def test_supersession_uses_only_the_controller_state_and_successor() -> None:
    first = policy.supersession({"state": "superseded", "superseded_by": SID})
    assert first is not None and first.successor == SID
    assert (
        policy.supersession({"state": "failed", "reason_code": "superseded-by-retry"})
        is None
    )
    assert (
        policy.supersession(
            {
                "state": "failed",
                "status_reason": f"Automatically reconciled by profile retry {SID}",
            }
        )
        is None
    )
    assert policy.supersession({"state": "succeeded"}) is None


def test_earlier_failures_recorded_for_a_superseded_load_are_pending_again(
    tmp_path, gateway
) -> None:
    recipes = [FakeRecipe("a"), FakeRecipe("b", ("m2",))]
    models = [FakeModel("m1"), FakeModel("m2")]
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    sweep.run()
    for slug, error in (
        ("a", f"Automatically reconciled by profile retry {SID}"),
        ("b", "container exited with code 1"),
    ):
        _entry(sweep, slug).update(
            status="failed", error=error, failure_class="start", phase="start"
        )
    sweep.state.data["schema"] = 1
    sweep.state.save()
    again, _, _ = make_sweep(tmp_path, recipes, models, fleet=fleet)
    again.run()
    assert _entry(again, "a")["status"] == "passed"
    assert _entry(again, "b")["status"] == "failed"


OLD_APP = "8cb1d3a7-0000-4000-8000-0000000000a1"
NEW_APP = "8cb1d3a7-0000-4000-8000-0000000000a2"


def _two_own_loads(sweep, observed: dict) -> dict:
    """An older load of ours and the later load of ours whose placement replaced it."""
    sweep.state.data["own_loads"] = [
        {"application_id": OLD_APP, "kind": "placement", "request_key": "key-old"},
        {"application_id": NEW_APP, "kind": "placement", "request_key": "key-new"},
    ]
    load = {"request_key": "key-old", "app_id": observed["id"], "state": "running"}
    sweep.state.data["loads"] = {"key-old": load}
    sweep.vk.run = lambda *_args, **_kwargs: Reply(("profile",), 0, observed, "")
    return load


@pytest.mark.parametrize(
    "observed",
    [
        # Hardware 2026-10-16: each placement of profile 10 superseded the previous
        # one, and every older load "followed" it to the newest application.
        {"id": OLD_APP, "state": "superseded", "superseded_by": NEW_APP},
        # A load already following another load's application settles on next look.
        {"id": NEW_APP, "state": "running"},
    ],
)
def test_a_load_without_bound_adoption_never_follows_another_own_load(
    tmp_path, gateway, observed
) -> None:
    sweep, _fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.preflight()
    sweep.refresh_catalog()
    recipe = sweep.recipes["vonk-forge/a"]
    sweep._open_slot(policy.Placement(recipe, ("spk_a",)), "key-old", sweep.clock.now())
    load = _two_own_loads(sweep, observed)

    assert sweep._poll_application(load)

    assert load["app_id"] != NEW_APP or observed["id"] == NEW_APP
    assert "key-old" not in sweep.state.data["loads"]
    assert sweep.state.data["apps"]["key-old"]["state"] == LifecycleState.SUPERSEDED
    sweep._advance_loading(recipe.key, sweep.state.slots[recipe.key], sweep.clock.now())
    assert sweep.log.entries()[0]["status"] == ReconciliationStatus.REQUEUED


def test_failed_application_does_not_probe_for_an_untyped_continuation(tmp_path):
    sweep, _, _ = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    calls = []

    def observe(*args, **kwargs):
        calls.append(args)
        return Reply(args, 0, {"id": OLD_APP, "state": "failed", "progress": {}}, "")

    sweep.vk.run = observe
    load = {"request_key": "key", "app_id": OLD_APP}
    sweep.state.data["loads"]["key"] = load
    assert sweep._poll_application(load)
    assert len(calls) == 1
    assert "key" not in sweep.state.data["loads"]
    assert sweep.state.data["apps"]["key"]["state"] == "failed"


def _supersede_on_load(fleet, monkeypatch) -> None:
    """Inject whole-fleet supersession while preserving unchanged children/runs."""
    original = fleet._load

    def load(number, data, args):
        previous = data.get("latest")
        reply = original(number, data, args)
        current = data.get("latest")
        if previous and previous != current and number == 10:
            fleet.apps[previous].update(
                state=LifecycleState.SUPERSEDED, superseded_by=current
            )
        return reply

    monkeypatch.setattr(fleet, "_load", load)


@pytest.mark.parametrize("phase", [SweepSlotPhase.LOADING, SweepSlotPhase.SMOKING])
@pytest.mark.parametrize("restart", [False, True])
def test_a_sibling_load_preserves_the_lane_and_its_pass(
    tmp_path, monkeypatch, phase, restart
) -> None:
    recipes = [
        FakeRecipe("a", local="cached", load_seconds=20, adapter="readiness"),
        FakeRecipe("b", local="cached", load_seconds=20, adapter="readiness"),
    ]
    models = [FakeModel("m1", local="cached")]
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models)
    _supersede_on_load(fleet, monkeypatch)
    sweep.preflight()
    sweep.refresh_catalog()
    a, b = recipes
    sweep._apply([policy.Placement(sweep.recipes[a.key], ("spk_a",))], clock.now())
    slot = sweep.state.slots[a.key]
    accepted = AcceptedRequest.read(slot)
    deadline = slot["deadline"]
    original_request = slot["request_key"]
    app = sweep.state.data["loads"][original_request]["app_id"]
    run = next(r["run_id"] for r in fleet.runs if r["recipe"] == a.key)
    pending: Future[SmokeResult] = Future()
    if phase == SweepSlotPhase.SMOKING:
        clock.sleep(20)
        sweep.advance_slots(clock.now())
        sweep.futures[a.key] = pending
    assert slot["phase"] == phase

    # Catalog drift cannot rebind the content already accepted by this lane.
    sweep.recipes[a.key] = replace(sweep.recipes[a.key], content_sha256="e" * 64)
    sweep._apply([policy.Placement(sweep.recipes[b.key], ("spk_b",))], clock.now())
    latest = fleet.profiles[10]["latest"]
    assert fleet.apps[app]["superseded_by"] == latest
    assert {x["assignment_name"] for x in fleet.profiles[10]["assignments"]} == {
        sweep.state.slots[a.key]["alias"],
        sweep.state.slots[b.key]["alias"],
    }
    assert next(r["run_id"] for r in fleet.runs if r["recipe"] == a.key) == run
    if restart:
        sweep.state.save()
        sweep, _, _ = make_sweep(tmp_path, recipes, models, fleet=fleet)
        sweep.preflight()
        sweep.refresh_catalog()
        if phase == SweepSlotPhase.SMOKING:
            # A resumed smoke has a fresh in-flight future.
            pending = Future()
            sweep.futures[a.key] = pending
    # Poll the successor first, including a terminal receipt, as can happen
    # after JSON state reload sorts request keys into a different order.
    sweep.state.data["loads"] = dict(reversed(list(sweep.state.data["loads"].items())))
    sweep.advance_slots(clock.now())
    assert sweep.state.slots[a.key]["phase"] == phase
    clock.sleep(20)
    sweep.advance_slots(clock.now())
    slot = sweep.state.slots[a.key]
    assert slot["phase"] == SweepSlotPhase.SMOKING
    assert slot["run_id"] == run
    assert slot["deadline"] == deadline
    assert AcceptedRequest.read(slot) == accepted
    if phase == SweepSlotPhase.SMOKING:
        assert sweep.futures[a.key] is pending
        pending.set_result(SmokeResult(True, "readiness-only"))
    sweep.advance_slots(clock.now())
    assert sweep.state.entry(a.key)["status"] == RecipeStatus.PASSED
    rows = [row for row in sweep.log.entries() if row["recipe"] == a.key]
    assert not any(row["status"] == ReconciliationStatus.REQUEUED for row in rows)
    passed = next(row for row in rows if row["status"] == RecipeStatus.PASSED)
    assert passed["request_key"] == original_request
    assert not any(args[:2] == ("profile", "cancel") for _, args in fleet.calls)


@pytest.mark.parametrize("count", [2, 3, 8, 17])
def test_alternating_lane_loads_make_progress_without_supersede_churn(
    tmp_path, monkeypatch, count
) -> None:
    recipes = [
        FakeRecipe(f"r{i}", local="cached", load_seconds=20, adapter="readiness")
        for i in range(count)
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")]
    )
    _supersede_on_load(fleet, monkeypatch)
    sweep.preflight()
    sweep.refresh_catalog()
    for index, recipe in enumerate(recipes):
        sweep._apply(
            [
                policy.Placement(
                    sweep.recipes[recipe.key], (("spk_a", "spk_b")[index % 2],)
                )
            ],
            clock.now(),
        )
        assert recipe.key in sweep.state.slots
        sweep.advance_slots(clock.now())
        clock.sleep(10)
        sweep.advance_slots(clock.now())
        sweep.advance_slots(clock.now())
        assert (
            sum(
                sweep.state.entry(r.key)["status"] == RecipeStatus.PASSED
                for r in recipes
            )
            == index
        )
        sweep.cleanup_finished()
    clock.sleep(10)
    sweep.advance_slots(clock.now())
    sweep.advance_slots(clock.now())
    assert all(
        sweep.state.entry(r.key)["status"] == RecipeStatus.PASSED for r in recipes
    )
    assert not any(
        row["status"] == ReconciliationStatus.REQUEUED for row in sweep.log.entries()
    )
    assert len(fleet.apps) == count
    assert not any(args[:2] == ("profile", "cancel") for _, args in fleet.calls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("application_id", SID),
        ("node_ids", ["spk_b"]),
        ("plan_digest", "invalid"),
        ("workload_intent_ordinal", 0),
        ("assignment_ids", []),
    ],
)
def test_unbound_adoption_cannot_submit_or_carry_a_lane(
    tmp_path, monkeypatch, field, value
) -> None:
    recipes = [FakeRecipe("a", local="cached"), FakeRecipe("b", local="cached")]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    a, b = recipes
    sweep._apply([policy.Placement(sweep.recipes[a.key], ("spk_a",))], clock.now())
    request = sweep.state.slots[a.key]["request_key"]
    original = fleet._load

    def corrupt_review(number, data, args):
        code, document = original(number, data, args)
        if "--review" in args:
            document["effects"]["adopted"][0][field] = value
        return code, document

    monkeypatch.setattr(fleet, "_load", corrupt_review)
    sweep._apply([policy.Placement(sweep.recipes[b.key], ("spk_b",))], clock.now())
    assert b.key not in sweep.state.slots
    assert sweep.state.slots[a.key]["request_key"] == request
    assert len(fleet.apps) == 1
    assert not list(sweep.log.entries())
