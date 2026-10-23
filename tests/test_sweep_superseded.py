"""A load the Controller replaced is followed or requeued, never recorded as a recipe failure."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts.sweep import SweepSlotPhase, SweepWaitCode, SweepWaitReason

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


def _pending_admission_edits_supersede(fleet, monkeypatch):
    """An unbound admission is invalidated by saved-profile edits, even rollback."""
    profile = fleet._profile
    load = fleet._load
    app_doc = fleet._app_doc
    edits = []

    def pending():
        return [
            app
            for app in fleet.apps.values()
            if app.get("blocked") and app["state"] == LifecycleState.RUNNING
        ]

    def edit(number, verb, args):
        if number == 10 and verb in ("add", "remove"):
            for app in pending():
                edits.append((fleet.clock.now(), app["id"]))
                app["state"] = LifecycleState.SUPERSEDED
        return profile(number, verb, args)

    def review(number, data, args):
        code, doc = load(number, data, args)
        if "--review" in args:
            ids = {app["id"] for app in pending()}
            doc["effects"]["adopted"] = [
                item
                for item in doc["effects"]["adopted"]
                if item["application_id"] not in ids
            ]
        return code, doc

    def document(app):
        doc = app_doc(app)
        if app in pending():
            doc["progress"].update(admission_pending=True, effects=[])
            doc["wait"] = {
                "reason": "profile.preparation_unavailable",
                "blockers": [
                    {
                        "code": "profile.preparation_unavailable",
                        "operation": {"kind": "recipe-build"},
                    }
                ],
            }
        return doc

    monkeypatch.setattr(fleet, "_profile", edit)
    monkeypatch.setattr(fleet, "_load", review)
    monkeypatch.setattr(fleet, "_app_doc", document)
    return edits


@pytest.mark.parametrize("binds", [False, True])
def test_pending_admission_protects_profile_then_releases_other_lane(
    tmp_path, monkeypatch, binds
):
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
    edits = _pending_admission_edits_supersede(fleet, monkeypatch)
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
    assert build_status(sweep)["profile_edit_wait"] is None


@pytest.mark.parametrize("cleanup", [False, True])
def test_pending_lane_reviews_once_per_tick_with_many_candidates(
    tmp_path, monkeypatch, cleanup
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
    edits = _pending_admission_edits_supersede(fleet, monkeypatch)
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
        assert len(reviews) == 1
        assert not edits
        assert len(sweep.state.slots) == 1
        clock.sleep(5)


@pytest.mark.parametrize(
    "review",
    [
        Reply(("profile",), 1, None, "partitioned"),
        Reply(("profile",), 0, {"unreadable": True}, ""),
    ],
)
def test_unreadable_profile_review_ends_at_pending_lane_bound(tmp_path, review):
    from vonk_forge_contracts.sweep import SweepWait

    from spark_sweep import verdict

    sweep, _, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    sweep.preflight()
    sweep.refresh_catalog()
    recipe = sweep.recipes["vonk-forge/a"]
    sweep._open_slot(policy.Placement(recipe, ("spk_a",)), "pending", clock.now())
    slot = sweep.state.slots[recipe.key]
    deadline = clock.now() + 30
    verdict.bounded_wait(
        SweepWait(reason=SweepWaitReason.ADMISSION),
        slot.setdefault("watch", {}),
        window=30,
        now=clock.now(),
    )
    sweep.vk.run = lambda *args, **kwargs: review
    assert not sweep._profile_edit_ready()
    wait = sweep.state.data["profile_edit_wait"]
    since = wait["wait_started_at"]
    clock.sleep(20)
    assert not sweep._profile_edit_ready()
    assert sweep.state.data["profile_edit_wait"] is wait
    assert wait["wait_started_at"] == since
    assert wait["wait_deadline"] == deadline
    clock.sleep(11)
    assert sweep._profile_edit_ready()
    assert not sweep.state.data.get("profile_edit_wait")


def test_profile_edit_wait_tracks_progress_after_dependency_wait(tmp_path):
    from vonk_forge_contracts.sweep import SweepWait

    from spark_sweep import verdict

    sweep, _, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    sweep.preflight()
    sweep.refresh_catalog()
    recipe = sweep.recipes["vonk-forge/a"]
    sweep._open_slot(policy.Placement(recipe, ("spk_a",)), "pending", clock.now())
    slot = sweep.state.slots[recipe.key]
    watch = slot.setdefault("watch", {})
    verdict.bounded_wait(
        SweepWait(reason=SweepWaitReason.ADMISSION), watch, window=30, now=clock.now()
    )
    sweep.vk.run = lambda *args, **kwargs: Reply(args, 0, {"effects": {}}, "")
    assert not sweep._profile_edit_ready()
    since = sweep.state.data["profile_edit_wait"]["wait_started_at"]
    clock.sleep(20)
    slot["copying"] = True
    verdict.progress_verdict(
        verdict.Signal(("copying",), None),
        watch,
        window=sweep._load_window(slot),
        now=clock.now(),
    )
    assert "wait" not in watch and "wait_deadline" in watch
    assert not sweep._profile_edit_ready()
    wait = sweep.state.data["profile_edit_wait"]
    assert wait["wait_started_at"] == since
    assert wait["wait_deadline"] == watch["progressed_at"] + sweep._load_window(slot)
    clock.sleep(wait["wait_deadline"] - clock.now())
    assert not sweep._profile_edit_ready()
    clock.sleep(1)
    assert sweep._profile_edit_ready()
    assert not sweep.state.data.get("profile_edit_wait")


def test_tick_clears_profile_edit_wait_after_last_loading_lane_ends(tmp_path):
    from vonk_forge_contracts.sweep import SweepWait

    from spark_sweep import verdict
    from spark_sweep.report import build_status, write_status

    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a", local="cached")], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    verdict.bounded_wait(
        SweepWait(reason=SweepWaitReason.ADMISSION),
        sweep.state.data.setdefault("profile_edit_wait", {}),
        window=30,
        now=clock.now(),
    )
    # A paused scheduler never reaches the edit guard. Status must still self-heal.
    sweep._observe_owner = lambda now: None
    sweep.owner_status = replace(sweep.owner_status, paused=True)
    clock.sleep(130)
    sweep._tick()
    assert not any(
        slot["phase"] == SweepSlotPhase.LOADING for slot in sweep.state.slots.values()
    )
    assert build_status(sweep)["profile_edit_wait"] is None
    write_status(sweep)
    assert "Placement waiting on" not in (tmp_path / "status.md").read_text()


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
    # One pre-edit adoption preview and one admission of the changed profile.
    assert len(reviews) == 2
    assert b.key in sweep.state.slots
    assert stale not in sweep.state.data["cleanup"]
