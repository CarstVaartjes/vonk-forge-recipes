"""A load that carries several recipes waits or fails for one of them: only that one is blamed."""

from __future__ import annotations

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from vonk_forge_contracts import QualificationBlame
from vonk_forge_contracts import QualificationCauseCode as Code

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
