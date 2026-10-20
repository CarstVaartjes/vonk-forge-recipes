"""A load that carries several recipes waits or fails for one of them: only that one is blamed."""

from __future__ import annotations

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep import policy
from spark_sweep.policy import TimeoutPolicy

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


def _trio(**held):
    recipes = [
        FakeRecipe("bad", ("m1",), local="cached", **held),
        FakeRecipe("good1", ("m1",), local="cached"),
        FakeRecipe("good2", ("m1",), local="cached"),
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
    return _make(
        tmp_path,
        gateway,
        _trio(blocked=True, blocker_code=PREP, **held),
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
