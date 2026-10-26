"""A load the Controller replaced is followed or requeued, never recorded as a recipe failure."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts.sweep import SweepSlotPhase, SweepWaitCode, SweepWaitReason

from spark_sweep import policy, verdict
from spark_sweep.identity import AcceptedRequest
from spark_sweep.lifecycle import LifecycleState, RecipeStatus, ReconciliationStatus
from spark_sweep.report import build_status, write_status
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


@pytest.mark.parametrize("binds", [False, True])
def test_pending_admission_protects_profile_then_releases_other_lane(tmp_path, binds):
    from spark_sweep.report import build_status, write_status

    a = FakeRecipe(
        "a",
        content="a" * 64,
        digests=("1" * 64,),
        local="cached",
        blocked=True,
        blocker_code="profile.preparation_unavailable",
    )
    b = FakeRecipe("b", content="b" * 64, digests=("1" * 64,), local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [a, b],
        [FakeModel("1" * 64, local="cached")],
        blocked_release_seconds=30,
    )
    edits = fleet.superseded_pending
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    assert not edits  # no profile mutation while the first admission is unbound
    slot = sweep.state.slots[a.key]
    request = slot["request_key"]
    application = sweep.state.data["loads"][request]["app_id"]
    sweep._tick()
    wait = build_status(sweep)["profile_edit_wait"]
    assert wait["wait"]["reason"] == SweepWaitReason.ADMISSION
    assert application in wait["waiting_on"]
    assert wait["wait_deadline"] == slot["watch"]["wait_deadline"]
    # The idle alarm names the hold placement applies, not a generic guess.
    assert application in sweep._lane_holds()["spk_b"]
    write_status(sweep)
    assert application in (tmp_path / "status.md").read_text()
    # Finished-lane cleanup must obey the same pre-edit boundary as placement.
    sweep.state.data["cleanup"] = {"old-lane": {"next_check": 0, "attempts": 0}}
    sweep.cleanup_finished()
    assert "old-lane" in sweep.state.data["cleanup"]
    for _ in range(4):
        clock.sleep(10)
        if binds:
            fleet.apps[application]["blocked"] = False
        sweep._tick()
        if b.key in sweep.state.slots:
            break
    assert all(at > slot["watch"]["wait_deadline"] for at, _ in edits)
    if binds:
        assert not edits
        assert fleet.apps[application]["state"] != LifecycleState.SUPERSEDED
    assert b.key in sweep.state.slots
    assert sweep.state.slots[b.key]["request_key"] != request
    assert sweep.state.data["loads"][sweep.state.slots[b.key]["request_key"]]["app_id"]
    if not binds:
        entry = sweep.state.entry(a.key)
        assert entry["wait_deferral"]["code"] == SweepWaitCode.BUDGET_EXHAUSTED
        assert entry["controller_wait"]["application_id"] == application
        assert entry.get("attempts", 0) == 0
        assert clock.now() <= slot["watch"]["wait_deadline"] + 10
        # B's request retired A's unbound admission: nothing waits on that receipt
        # any more (B's load is not A's), so A is planned again at a named time
        # rather than held for ever.
        for _ in range(4):
            if not sweep.controller_wait_holds()[a.key].live:
                break
            clock.sleep(fleet.admission_retry_seconds)
            sweep._refresh_controller_waits(clock.now())
        assert [app for _, app in edits] == [application]
        assert fleet.apps[application]["state"] == LifecycleState.SUPERSEDED
        hold = sweep.controller_wait_holds()[a.key]
        assert hold.application_id == application
        assert not hold.live and hold.not_before is not None
        assert a.key not in {r.key for r in sweep.ready_candidates()}
        clock.t = hold.not_before
        assert a.key in {r.key for r in sweep.ready_candidates()}
    assert build_status(sweep)["profile_edit_wait"] is None


@pytest.mark.parametrize("cleanup", [False, True])
def test_pending_lane_holds_edits_without_reviews_with_many_candidates(
    tmp_path, cleanup
):
    recipes = [
        FakeRecipe(
            "a",
            local="cached",
            blocked=True,
            blocker_code="profile.preparation_unavailable",
        ),
        *(FakeRecipe(f"r{i}", local="cached") for i in range(5)),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached")],
        blocked_release_seconds=30,
    )
    edits = fleet.superseded_pending
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    if cleanup:
        sweep.state.data["cleanup"] = {"old-lane": {"next_check": 0, "attempts": 0}}
    for _ in range(2):
        fleet.calls.clear()
        sweep._tick()
        reviews = [
            args
            for number, args in fleet.calls
            if number == 10 and args[:2] == ("profile", "load") and "--review" in args
        ]
        # The lane's own receipt shows the unbound admission; no preview is needed.
        assert not reviews
        assert not edits
        assert len(sweep.state.slots) == 1
        clock.sleep(5)


def _pending_lane(tmp_path, **overrides):
    recipes = [
        FakeRecipe(
            "a",
            local="cached",
            blocked=True,
            blocker_code="profile.preparation_unavailable",
        ),
        FakeRecipe("b", local="cached"),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached")],
        blocked_release_seconds=30,
        only=("a",),
        **overrides,
    )
    edits = fleet.superseded_pending
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    sweep._tick()
    slot = sweep.state.slots[recipes[0].key]
    application = sweep.state.data["loads"][slot["request_key"]]["app_id"]
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    return sweep, fleet, clock, slot, application, edits


def test_decided_clearing_waits_for_pending_application_without_review(tmp_path):
    sweep, fleet, clock, slot, application, edits = _pending_lane(tmp_path)
    # The capacity review already decided to clear. No cleanup receipt exists yet.
    sweep.state.data["cleared_for"] = ["vonk-forge/b"]
    sweep._request_clearing(clock.now())
    for _ in range(2):
        fleet.calls.clear()
        sweep._tick()
        reviews = [
            args
            for number, args in fleet.calls
            if number == 10 and args[:2] == ("profile", "load") and "--review" in args
        ]
        assert not reviews
        status = build_status(sweep)
        wait = status.get("profile_edit_wait")
        assert wait is not None
        assert wait["wait"]["reason"] == SweepWaitReason.ADMISSION
        assert application in sweep.state.data["clearing_refusal"] == wait["waiting_on"]
        assert wait["wait_deadline"] == slot["watch"]["wait_deadline"]
        write_status(sweep)
        text = (tmp_path / "status.md").read_text()
        assert application in text
        assert datetime.fromtimestamp(wait["wait_deadline"], UTC).isoformat() in text
        assert not edits
        clock.sleep(5)
        sweep._request_clearing(clock.now())


@pytest.mark.parametrize("idle", ["pause", "empty", "takeover"])
def test_tick_clears_profile_edit_wait_when_placement_does_not_run(
    tmp_path, monkeypatch, idle
):
    sweep, fleet, clock, slot, application, _ = _pending_lane(tmp_path)
    sweep._tick()
    assert build_status(sweep).get("profile_edit_wait")
    fleet.apps[application]["blocked"] = False
    if idle == "pause":
        monkeypatch.setattr(sweep, "_observe_owner", lambda now: None)
        sweep.owner_status = replace(sweep.owner_status, paused=True)
    elif idle == "empty":
        sweep.cfg = replace(sweep.cfg, only=("a",))
    else:
        monkeypatch.setattr(sweep, "_take_over", lambda *args, **kwargs: False)
    clock.sleep(5)
    # A cleanup still needs a fresh guard when scheduling has no candidates.
    if idle != "pause":
        sweep.state.data["cleanup"] = {"old-lane": {"next_check": 0, "attempts": 0}}
    sweep._tick()
    assert slot["phase"] == SweepSlotPhase.LOADING
    assert "vonk-forge/b" not in sweep.state.slots
    assert build_status(sweep)["profile_edit_wait"] is None
    if idle != "pause":
        assert not sweep.state.data["cleanup"]
    write_status(sweep)
    assert "Placement waiting on" not in (tmp_path / "status.md").read_text()


@pytest.mark.parametrize("unreadable", [False, True])
def test_partitioned_review_heals_and_admits_within_lane_bound(
    tmp_path, monkeypatch, unreadable
):
    sweep, fleet, clock, slot, application, edits = _pending_lane(tmp_path)
    deadline = slot["watch"]["wait_deadline"]
    original = fleet._load
    partitioned = True

    def review(number, data, args):
        if number == 10 and "--review" in args and partitioned:
            return (0, {"unreadable": True}) if unreadable else (1, None)
        return original(number, data, args)

    monkeypatch.setattr(fleet, "_load", review)
    for _ in range(2):
        clock.sleep(5)
        sweep._tick()
        wait = build_status(sweep).get("profile_edit_wait")
        assert wait is not None
        assert application in wait["waiting_on"]
        assert wait["wait_deadline"] == deadline
        assert "vonk-forge/b" not in sweep.state.slots
        assert not edits
    partitioned = False
    fleet.apps[application]["blocked"] = False
    clock.sleep(5)
    sweep._tick()
    assert "vonk-forge/b" in sweep.state.slots
    assert clock.now() < deadline
    assert build_status(sweep)["profile_edit_wait"] is None
    assert not edits


def test_stale_assignment_prune_reuses_profile_edit_readiness(tmp_path):
    recipes = [FakeRecipe(name, local="cached") for name in ("a", "b")]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    a, b = (sweep.recipes[recipe.key] for recipe in recipes)
    sweep._apply([policy.Placement(a, ("spk_a",))], clock.now())
    stale = "sweep-stale"
    sweep._remember_own(stale)
    fleet.profiles[10]["assignments"].append(
        {**fleet.profiles[10]["assignments"][0], "assignment_name": stale}
    )
    fleet.calls.clear()
    sweep._apply([policy.Placement(b, ("spk_b",))], clock.now())
    reviews = [
        args
        for number, args in fleet.calls
        if number == 10 and args[:2] == ("profile", "load") and "--review" in args
    ]
    # A bound lane needs no pre-edit preview: one review admits the changed profile.
    assert len(reviews) == 1
    assert b.key in sweep.state.slots
    assert stale not in sweep.state.data["cleanup"]


def test_bound_copying_lane_does_not_hold_the_free_lane(tmp_path):
    a = FakeRecipe("a", ("m1",), local="cached", copy_seconds=3000)
    b = FakeRecipe("b", ("m2",), local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [a, b],
        [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")],
        only=("a",),
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    request = sweep.state.slots[a.key]["request_key"]
    application = sweep.state.data["loads"][request]["app_id"]
    clock.sleep(20)
    sweep._tick()
    assert sweep.state.slots[a.key]["copying"]
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    clock.sleep(20)
    sweep._tick()
    # The other Spark loads while the bound copy keeps progressing and is adopted.
    assert b.key in sweep.state.slots
    assert sweep.state.slots[a.key]["phase"] == SweepSlotPhase.LOADING
    assert sweep.state.slots[a.key]["copying"]
    load = sweep.state.data["loads"][sweep.state.slots[b.key]["request_key"]]
    assert load["adopted_requests"] == [request]
    assert fleet.apps[application]["state"] == LifecycleState.RUNNING
    assert build_status(sweep)["profile_edit_wait"] is None


def test_free_lane_reserved_for_a_dual_names_the_reservation(tmp_path):
    single = FakeRecipe("s", ("m1",), local="cached", load_seconds=5000)
    dual = FakeRecipe("d", ("m2",), local="cached", node_count=2)
    sweep, _, clock = make_sweep(
        tmp_path,
        [single, dual],
        [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")],
        only=("s",),
        idle_alert_seconds=60,
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    alias = sweep.state.slots[single.key]["alias"]
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    for _ in range(9):
        clock.sleep(10)
        sweep._tick()
    assert dual.key not in sweep.state.slots
    status = build_status(sweep)
    assert (
        dual.key in status["dual_reservation"] and alias in status["dual_reservation"]
    )
    (alert,) = status["alerts"]
    assert f"reserved for dual {dual.key} behind lane {alias}" in alert
    write_status(sweep)
    assert f"reserved for dual {dual.key}" in (tmp_path / "status.md").read_text()


@pytest.mark.parametrize(
    ("state", "operation", "pending", "unbound"),
    [
        # Only a pending admission is superseded by a save (pending_admission.py).
        (LifecycleState.QUEUED, None, True, True),
        (LifecycleState.NEEDS_OPERATOR, None, True, True),
        # Bound: a newer decision adopts its unchanged assignments.
        (LifecycleState.QUEUED, None, False, False),
        (LifecycleState.RUNNING, "8cb1d3a7-0000-4000-8000-000000000002", False, False),
    ],
)
def test_admission_unbound_mirrors_what_a_profile_save_supersedes(
    state, operation, pending, unbound
):
    document = {
        "id": "8cb1d3a7-0000-4000-8000-000000000001",
        "state": state,
        "current_step": 0,
        "current_operation_id": operation,
        "progress": {"admission_pending": pending},
    }
    assert verdict.admission_unbound(document) is unbound


def test_unadopted_bound_lane_names_its_hold_and_stops_the_pass(tmp_path, monkeypatch):
    a = FakeRecipe("a", ("m1",), local="cached", load_seconds=5000)
    rest = [FakeRecipe(f"r{i}", ("m2",), local="cached") for i in range(5)]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [a, *rest],
        [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")],
        only=("a",),
        idle_alert_seconds=30,
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    alias = sweep.state.slots[a.key]["alias"]
    original = fleet._load

    def unadopted(number, data, args):
        code, document = original(number, data, args)
        if "--review" in args:
            document["effects"]["adopted"] = []
        return code, document

    monkeypatch.setattr(fleet, "_load", unadopted)
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    for _ in range(5):
        clock.sleep(10)
        fleet.calls.clear()
        sweep._tick()
        adds = [
            args
            for number, args in fleet.calls
            if number == sweep.cfg.sweep_profile and args[:2] == ("profile", "add")
        ]
        # One candidate pays add, review and remove; the pass then stops on the
        # named hold instead of repeating it for every other candidate.
        assert len(adds) <= 1
        assert set(sweep.state.slots) == {a.key}
    hold = sweep._lane_holds()["spk_b"]
    assert f"loading lane(s) {alias} not adopted" in hold and "retry at" in hold
    (alert,) = build_status(sweep)["alerts"]
    assert f"{alias} not adopted by the review of the edited profile" in alert
    # Adoption binds again: the free Spark is placed in the next pass.
    monkeypatch.setattr(fleet, "_load", original)
    clock.sleep(sweep.cfg.retry_delay)
    sweep._tick()
    assert len(sweep.state.slots) == 2
    assert "not adopted" not in sweep._lane_holds()["spk_b"]


def test_a_followed_successor_holds_edits_until_its_admission_is_observed(tmp_path):
    a = FakeRecipe("a", ("m1",), local="cached", copy_seconds=3000)
    b = FakeRecipe("b", ("m2",), local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [a, b],
        [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")],
        only=("a",),
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    clock.sleep(20)
    sweep._tick()
    slot = sweep.state.slots[a.key]
    load = sweep.state.data["loads"][slot["request_key"]]
    assert slot["copying"] and load["admission_unbound"] is False
    # The Controller continues A as a successor the sweep has not observed yet.
    fleet.supersede = ["retry"]
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    clock.sleep(20)
    sweep._tick()
    assert fleet.apps[load["app_id"]].get("successor_of")
    assert b.key not in sweep.state.slots
    wait = build_status(sweep)["profile_edit_wait"]
    # No dependency wait: the lane's own stall deadline bounds the hold.
    assert "wait" not in slot["watch"]
    assert wait["wait_deadline"] == (
        slot["watch"]["progressed_at"] + sweep._load_window(slot)
    )
    clock.sleep(20)
    sweep._tick()
    # Observed bound: the successor is carried by adoption and B is placed.
    assert load["admission_unbound"] is False
    assert b.key in sweep.state.slots
    assert build_status(sweep)["profile_edit_wait"] is None
