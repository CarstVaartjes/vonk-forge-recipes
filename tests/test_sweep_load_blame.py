"""A load that carries several recipes waits or fails for one of them: only that one is blamed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from vonk_forge_contracts import QualificationBlame
from vonk_forge_contracts import QualificationCauseCode as Code
from vonk_forge_contracts.sweep import WorkloadExitCause

from spark_sweep import policy
from spark_sweep.lifecycle import RecipeStatus
from spark_sweep.policy import TimeoutPolicy
from spark_sweep.qualification import from_history
from spark_sweep.verdict import Disposition, disposition

PREP = "profile.preparation_unavailable"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _timeouts(seconds: float = 600.0) -> TimeoutPolicy:
    return TimeoutPolicy(
        first_start_seconds=seconds, floor_seconds=seconds, cap_seconds=seconds
    )


# A held admission's wait is the Controller's: its expiry records the content identity.
DIGEST = "1" * 64


def _trio(**held):
    recipes = [
        FakeRecipe("bad", (DIGEST,), content="b" * 64, local="cached", **held),
        FakeRecipe("good1", (DIGEST,), content="c" * 64, local="cached"),
        FakeRecipe("good2", (DIGEST,), content="d" * 64, local="cached"),
    ]
    return recipes


def _make(tmp_path, gateway, recipes, **overrides):
    return make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached")],
        gateway=gateway,
        timeouts=_timeouts(),
        **overrides,
    )


def _status(sweep, slug):
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _results(sweep):
    return [e["status"] for e in sweep.state.recipes.values()]


def _events(sweep):
    return " | ".join(str(e) for e in sweep.state.data.get("events", []))


def test_blockers_carry_the_identifiers_they_name() -> None:
    document = {
        "progress": {
            "blockers": [
                {
                    "code": PREP,
                    "detail": "x",
                    "severity": "error",
                    "assignment": {"alias": "dg", "node_id": "spk_1"},
                    "recipes": ["vonk-forge/dg-single"],
                }
            ]
        }
    }
    (blocker,) = policy.admission_blockers(document)
    assert {"dg", "spk_1", "vonk-forge/dg-single"} <= set(blocker["refs"])


def test_blamed_prefers_the_recipe_over_its_spark() -> None:
    lanes = {
        "vonk-forge/a": {
            "names": ["alias-a", "vonk-forge/a", "a"],
            "places": ["spk_1"],
        },
        "vonk-forge/b": {
            "names": ["alias-b", "vonk-forge/b", "b"],
            "places": ["spk_1"],
        },
    }
    by_alias = {"refs": ["alias-b", "spk_1"], "detail": ""}
    assert policy.blamed([by_alias], lanes) == {"vonk-forge/b"}
    by_node = {"refs": ["spk_1"], "detail": ""}
    assert policy.blamed([by_node], lanes) == {"vonk-forge/a", "vonk-forge/b"}
    in_text = {"refs": [], "detail": "image for vonk-forge/a is missing"}
    assert policy.blamed([in_text], lanes) == {"vonk-forge/a"}
    assert policy.blamed([{"refs": [], "detail": "nothing here"}], lanes) == set()


def _make_held(tmp_path, gateway, release, **held):
    return make_sweep(
        tmp_path,
        _trio(blocked=True, blocker_code=PREP, **held),
        [FakeModel(DIGEST, local="cached")],
        gateway=gateway,
        timeouts=_timeouts(),
        blocked_release_seconds=release,
    )


def test_a_hold_is_waited_for_and_every_recipe_then_passes(tmp_path, gateway) -> None:
    sweep, _, _ = _make_held(tmp_path, gateway, release=10**9, blocked_for=3000)
    sweep.run()
    assert _results(sweep) == ["passed", "passed", "passed"]
    assert not any(e.get("failure_class") for e in sweep.state.recipes.values())


def test_the_ready_recipes_are_released_early_and_pass(tmp_path, gateway) -> None:
    sweep, _, _ = _make_held(tmp_path, gateway, release=60, blocked_for=3000)
    sweep.run()
    # Independent applications let the healthy recipes finish before preparation
    # ends; the held recipe keeps its own attempt and later passes.
    assert _status(sweep, "good1")["status"] == "passed"
    assert _status(sweep, "good2")["status"] == "passed"
    bad = _status(sweep, "bad")
    assert bad["status"] == "passed" and bad["attempts"] == 1
    assert all(
        _status(sweep, key)["finished_at"] < bad["finished_at"]
        for key in ("good1", "good2")
    )


def test_an_unnamed_hold_blames_nobody_and_everyone_waits(tmp_path, gateway) -> None:
    sweep, _, _ = _make_held(
        tmp_path, gateway, release=10**9, blocker_unnamed=True, blocked_for=3000
    )
    sweep.run()
    assert _results(sweep) == ["passed", "passed", "passed"]


def test_the_controllers_own_end_of_a_hold_names_the_recipe(tmp_path, gateway) -> None:
    sweep, _, _ = _make_held(
        tmp_path,
        gateway,
        release=10**9,
        blocked_for=3000,
        blocked_then="admission refused: does not fit",
    )
    sweep.run()
    assert _status(sweep, "bad")["status"] == "deferred"
    assert "admission refused" in _status(sweep, "bad")["error"]
    assert _status(sweep, "good1")["status"] == "passed"
    assert _status(sweep, "good2")["status"] == "passed"


def test_an_application_failure_names_the_failing_assignment(tmp_path, gateway) -> None:
    sweep, _, _ = _make(
        tmp_path,
        gateway,
        [
            FakeRecipe("bad", ("m1",), fail_load="boom", fail_phase="runtime-install"),
            FakeRecipe("good1", ("m1",), local="cached"),
            FakeRecipe("good2", ("m1",), local="cached"),
        ],
    )
    sweep.run()
    assert _status(sweep, "bad")["status"] == "deferred"
    assert _status(sweep, "good1")["status"] == "passed"
    assert _status(sweep, "good2")["status"] == "passed"


@pytest.mark.parametrize(
    ("summary", "expected_code"),
    [
        ("exit_code=2 oom_killed=false exit_cause=bad_arguments", Code.RUNTIME_EXIT),
        (
            (
                "exit_code=1 oom_killed=false exit_cause=python_exception "
                'exception="RuntimeError: engine adapter initialization failed"'
            ),
            Code.RUNTIME_EXIT,
        ),
        ("exit_code=127 exit_cause=missing_file", Code.RUNTIME_EXIT),
        ("exit_code=1 exit_cause=unsupported_gpu_arch", Code.RUNTIME_EXIT),
        ("exit_code=1 exit_cause=image_arch_mismatch", Code.RUNTIME_EXIT),
        ("exit_code=137 oom_killed=true exit_cause=oom_killed", Code.RUNTIME_OOM),
        (
            (
                "reason=workload.host_memory_exhausted exit_cause=host_memory_exhausted "
                "mem_available_bytes=1 memory_full_avg10=99 "
                "trigger=nvidia_allocation_failure sustained_ms=3000"
            ),
            Code.RUNTIME_OOM,
        ),
    ],
)
def test_controller_exit_summary_blames_only_its_recipe(
    tmp_path, gateway, summary, expected_code
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [
            FakeRecipe(
                "bad",
                ("a" * 64,),
                content="c" * 64,
                fail_load=f"runtime_process_exited: {summary}",
                fail_phase="start",
            ),
            FakeRecipe("good1", ("a" * 64,), content="d" * 64, local="cached"),
        ],
        [FakeModel("a" * 64, local="cached")],
        gateway=gateway,
    )
    original = fleet._app_doc

    def controller_reply(app):
        document = original(app)
        # The real switch adapter carries a reason, not a recipe failure code.
        for failure in document["progress"]["switch_adapter"]["assignment_failures"]:
            failure.pop("code", None)
        return document

    fleet._app_doc = controller_reply
    sweep.run()
    failed = _status(sweep, "bad")
    assert failed["status"] == RecipeStatus.FAILED
    assert failed["code"] == expected_code
    assert failed["attempts"] == 1
    assert summary in failed["error"]
    assert _status(sweep, "good1")["status"] == RecipeStatus.PASSED
    row = next(
        row for row in sweep.log.entries() if row.get("status") == RecipeStatus.FAILED
    )
    record = from_history(row)
    assert record is not None
    assert record.blame == QualificationBlame.RECIPE
    assert record.cause_code == expected_code


@pytest.mark.parametrize(
    ("phase", "code", "summary"),
    [
        ("start", "application.failed", "exit_code=1 exit_cause=unclassified"),
        ("start", "application.failed", "runtime_process_exited: exit_code=1"),
        ("start", "application.failed", "exit_code=1 exit_cause=future_unknown_cause"),
        ("start", "application.failed", "exit_code=139 exit_cause=crashed_by_signal"),
        ("start", "application.failed", "exit_code=1 exit_cause=start_refused"),
        (
            "start",
            "application.failed",
            "exit_cause=host_memory_exhausted trigger=sustained_full_psi",
        ),
        (
            "start",
            "application.failed",
            "exit_cause=host_memory_exhausted trigger=sustained_memavailable_exhaustion",
        ),
        (
            "start",
            "application.failed",
            'exit_cause=unclassified exception="exit_cause=bad_arguments"',
        ),
        (
            "start",
            "application.failed",
            "exit_cause=unclassified exit_cause=bad_arguments",
        ),
        ("install", "application.failed", "exit_cause=python_exception"),
        ("review", "application.failed", "exit_cause=bad_arguments"),
        ("start", "controller.transport_timeout", "exit_cause=python_exception"),
        ("start", "storage.insufficient_capacity", "exit_cause=python_exception"),
        ("start", "runtime_unavailable", "exit_cause=python_exception"),
    ],
)
def test_exit_facts_do_not_blame_platform_failures_on_recipes(
    phase, code, summary
) -> None:
    failure = policy.classify_application_failure(phase, code, summary)
    assert failure.code not in policy.RECIPE_FAILURE_CODES
    if code == "application.failed" and phase == "start":
        assert disposition(failure) == Disposition.OBSERVE


def test_unknown_exit_retries_without_recipe_blame_and_recovers(
    tmp_path, gateway
) -> None:
    recipe = FakeRecipe(
        "bad",
        ("a" * 64,),
        content="c" * 64,
        fail_load="runtime_process_exited: exit_cause=unclassified",
        fail_phase="start",
    )
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [recipe],
        [FakeModel("a" * 64, local="cached")],
        gateway=gateway,
    )

    def recover(_now):
        if _status(sweep, "bad").get("attempts", 0) >= 2:
            fleet.recipes[recipe.key].fail_load = None

    clock.hooks.append(recover)
    sweep.run()
    assert _status(sweep, "bad")["status"] == RecipeStatus.PASSED
    rows = [
        row
        for row in sweep.log.entries()
        if row.get("disposition") == Disposition.OBSERVE
    ]
    assert len(rows) == 2
    assert all(row["code"] not in policy.RECIPE_FAILURE_CODES for row in rows)
    # Retry observations are not recipe verdicts or public hardware failures.
    assert all(from_history(row) is None for row in rows)


@pytest.mark.parametrize(
    ("summary", "expected_code"),
    [
        ("exit_code=2 exit_cause=bad_arguments", Code.RUNTIME_EXIT),
        ("exit_code=1 exit_cause=python_exception", Code.RUNTIME_EXIT),
        (
            "exit_cause=host_memory_exhausted trigger=nvidia_allocation_failure",
            Code.RUNTIME_OOM,
        ),
    ],
)
def test_real_assignment_failure_shape_uses_its_exit_reason(
    tmp_path, gateway, summary, expected_code
) -> None:
    recipe = FakeRecipe(
        "bad",
        ("a" * 64,),
        content="c" * 64,
        fail_load=f"the workload process exited: runtime_process_exited: {summary}",
        fail_phase="start",
    )
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [recipe],
        [FakeModel("a" * 64, local="cached")],
        gateway=gateway,
    )
    original = fleet._app_doc

    def controller_reply(app):
        document = original(app)
        failures = document["progress"]["switch_adapter"]["assignment_failures"]
        # FleetProfileAssignmentFailure's exact projection: no code or alias.
        for failure in failures:
            reason = failure["reason"]
            failure.clear()
            failure.update(
                queue_index=0,
                assignment_id="00000000-0000-4000-8000-000000000001",
                operation_id="00000000-0000-4000-8000-000000000002",
                reason=reason,
                terminal=True,
            )
        return document

    fleet._app_doc = controller_reply
    sweep.run()
    failed = _status(sweep, "bad")
    assert failed["status"] == RecipeStatus.FAILED
    assert failed["code"] == expected_code
    assert failed["attempts"] == 1
    row = next(
        row for row in sweep.log.entries() if row.get("status") == RecipeStatus.FAILED
    )
    record = from_history(row)
    assert record is not None and record.blame == QualificationBlame.RECIPE


# The Controller's FleetProfileAssignmentFailure, field for field
# (control/src/vonk_control/fleet_profile_contract/switch_state.py).
ASSIGNMENT_FAILURE_FIELDS = {
    "queue_index",
    "assignment_id",
    "operation_id",
    "reason",
    "terminal",
}


# FleetProfileApplicationView (fleet_profile_contract/review.py), field for field.
APPLICATION_VIEW_FIELDS = {
    "id",
    "request_key",
    "profile_id",
    "profile_digest",
    "plan_digest",
    "attempt",
    "retry_of_application_id",
    "superseded_by",
    "reason_code",
    "state",
    "current_step",
    "total_steps",
    "current_operation_id",
    "status_reason",
    "progress",
    "cancellation",
    "result",
    "projection_issue",
    "blockers",
    "wait",
    "next_attempt_at",
    "created_at",
    "updated_at",
}


# A real runtime exit: the Controller's reason carries the helper's exit facts and, like the
# real one, names no recipe or alias.
EXIT = "runtime_process_exited: exit_code=2 exit_cause=bad_arguments exception={}"


def _shared(tmp_path, gateway, bad, other):
    """Two lanes in one application: the whole-profile load adopted the first lane's load."""
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [bad, other],
        [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")],
        gateway=gateway,
        only=("bad",),
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    clock.sleep(20)
    sweep._tick()
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    clock.sleep(20)
    sweep._tick()
    keys = [bad.key, other.key]
    request = sweep.state.slots[keys[1]]["request_key"]
    assert sweep.state.data["loads"][request]["adopted_requests"]
    # What polling the whole-profile application does for every adopted lane.
    sweep._carry_loading_slots(request)
    assert len({sweep.state.slots[k]["request_key"] for k in keys}) == 1
    return sweep, fleet, clock


def _settle(sweep, clock):
    for _ in range(400):
        clock.sleep(20)
        sweep._tick()
        if all(
            e["status"] in (RecipeStatus.FAILED, RecipeStatus.PASSED)
            for e in sweep.state.recipes.values()
        ):
            return


def _recipes(first, second=None):
    """Two lanes; the second fails with its own reason, or runs on and passes."""
    return tuple(
        FakeRecipe(
            slug,
            (model,),
            local="cached",
            load_seconds=300,
            **({"fail_load": reason, "fail_phase": "start"} if reason else {}),
        )
        for slug, model, reason in (("bad", "m1", first), ("good", "m2", second))
    )


FIXTURE = Path(__file__).parent / "fixtures" / "sweep-verdicts"


def test_the_real_failed_application_resolves_each_failure_to_its_lane() -> None:
    """Application 11a29362 (2026-10-28): one assignment failure, one adopted lane's run effect."""
    document = json.loads((FIXTURE / "failed-adopted-lane.json").read_text())
    own, adopted = policy.assignment_failures(document)
    assert (own.alias, own.from_effect) == ("glm-5.3-flash-d3490", False)
    assert "exit_cause=oom_killed" in own.reason
    assert own.node_ids == ("spk_9a86fdbab116442ab6707bf4181a3c1c",)
    assert (adopted.alias, adopted.from_effect) == ("glm-5.3-flash", True)
    assert adopted.failure_code == "workload.host_memory_exhausted"
    assert "host_memory_exhausted" in adopted.reason
    assert adopted.node_ids == ("spk_2818d189042b4c77aefa7796f4befd23",)
    lanes = {
        "own": {"names": [own.alias], "places": list(own.node_ids)},
        "adopted": {"names": [adopted.alias], "places": list(adopted.node_ids)},
    }
    assert policy.blamed([f.as_state() for f in (own, adopted)], lanes) == {
        "own",
        "adopted",
    }
    assert policy.blamed([own.as_state()], lanes) == {"own"}


def test_the_fake_matches_the_real_documents_shape(tmp_path, gateway) -> None:
    real = json.loads((FIXTURE / "failed-adopted-lane.json").read_text())["progress"]
    sweep, fleet, clock = _shared(tmp_path, gateway, *_recipes(EXIT.format("fault1")))
    _settle(sweep, clock)
    app = next(a for a in fleet.apps.values() if a.get("failures"))
    application = fleet._app_doc(app)
    assert set(application) <= APPLICATION_VIEW_FIELDS
    fake = application["progress"]
    (real_failure,) = real["switch_adapter"]["assignment_failures"]
    failures = fake["switch_adapter"]["assignment_failures"]
    assert failures
    for failure in failures:
        assert set(failure) == set(real_failure) == ASSIGNMENT_FAILURE_FIELDS
    (real_owner, _) = real["intended_profile"]["assignments"]
    for owner in fake["intended_profile"]["assignments"]:
        assert set(owner) <= set(real_owner)
        assert {k for k in owner["nodes"][0]} <= set(real_owner["nodes"][0])
    # Both directions: what the sweep reads from the real document exists in the fake's.
    for failure in policy.assignment_failures({"progress": fake}):
        assert failure.alias and failure.node_ids


def test_each_failing_lane_of_a_shared_application_gets_its_own_verdict(
    tmp_path, gateway
) -> None:
    sweep, _, clock = _shared(
        tmp_path, gateway, *_recipes(EXIT.format("fault1"), EXIT.format("fault2"))
    )
    _settle(sweep, clock)
    for slug, fault in (("bad", "fault1"), ("good", "fault2")):
        entry = _status(sweep, slug)
        assert entry["status"] == RecipeStatus.FAILED
        assert entry["attempts"] == 1
        assert entry["code"] == Code.RUNTIME_EXIT
        assert f"exception={fault}" in entry["error"]


def test_an_adopted_lanes_own_failed_run_effect_is_its_own_result(
    tmp_path, gateway
) -> None:
    sweep, fleet, clock = _shared(
        tmp_path, gateway, *_recipes(EXIT.format("fault1"), EXIT.format("fault2"))
    )
    original = fleet._app_doc

    def controller_reply(app):
        document = original(app)
        progress = document["progress"]
        owners = {a["id"]: a for a in progress["intended_profile"]["assignments"]}
        failures = progress["switch_adapter"]["assignment_failures"]
        # The adopted lane failed under an earlier application: only its run effect says so.
        for failure in [f for f in failures if "fault2" in f["reason"]]:
            failures.remove(failure)
            owner = owners[failure["assignment_id"]]
            progress["effects"] = [
                {
                    "effect_id": "e1",
                    "application_id": "00000000-0000-4000-8000-0000000000aa",
                    "kind": "run",
                    "target_id": failure["assignment_id"],
                    "node_ids": [n["node_id"] for n in owner["nodes"]],
                    "state": "failed",
                    "progress": {"members": [{"error": failure["reason"]}]},
                    "result": {
                        "run_switch": {"failure_code": "workload.host_memory_exhausted"}
                    },
                }
            ]
        return document

    fleet._app_doc = controller_reply
    _settle(sweep, clock)
    entry = _status(sweep, "good")
    assert entry["status"] == RecipeStatus.FAILED and entry["attempts"] == 1
    assert "exception=fault2" in entry["error"]
    assert _status(sweep, "bad")["attempts"] == 1


@pytest.mark.parametrize(
    ("cause", "expected_code"),
    [
        (WorkloadExitCause.BAD_ARGUMENTS, Code.RUNTIME_EXIT),
        (WorkloadExitCause.OOM_KILLED, Code.RUNTIME_OOM),
    ],
)
def test_assignment_reason_alone_supplies_the_lane_code(
    tmp_path, gateway, cause, expected_code
) -> None:
    bad, good = _recipes(f"runtime_process_exited: exit_cause={cause}")
    sweep, fleet, clock = _shared(tmp_path, gateway, bad, good)
    original = fleet._app_doc

    def controller_reply(app):
        # No aggregate cause: the only failure facts belong to the assignment.
        document = {
            key: value
            for key, value in original(app).items()
            if key in APPLICATION_VIEW_FIELDS
        }
        document["status_reason"] = None
        return document

    fleet._app_doc = controller_reply
    observed = []
    fail_slot = sweep._fail_slot

    def record_failure(key, slot, failure, app):
        observed.append(dict(app))
        return fail_slot(key, slot, failure, app)

    sweep._fail_slot = record_failure
    _settle(sweep, clock)
    assert observed
    assert _status(sweep, "bad")["code"] == expected_code
    assert _status(sweep, "good")["status"] == RecipeStatus.PASSED
    assert all("failure_code" not in app for app in observed)
