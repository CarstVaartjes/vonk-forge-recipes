"""A load that carries several recipes is held back or fails for one of them: only that one is blamed."""

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


def test_a_stall_names_one_recipe_and_the_others_are_collateral(
    tmp_path, gateway
) -> None:
    sweep, _, _ = _make_stalled(tmp_path, gateway, release=10**9)
    sweep.run()
    bad = _status(sweep, "bad")
    assert (bad["status"], bad["failure_class"]) == ("deferred", "admission-stalled")
    for slug in ("good1", "good2"):
        entry = _status(sweep, slug)
        assert entry["status"] != "failed", slug
        assert not entry.get("failure_class")
    assert "collateral of vonk-forge/bad" in _events(sweep)


def _make_stalled(tmp_path, gateway, release, **held):
    return _make(
        tmp_path,
        gateway,
        _trio(blocked=True, blocker_code=PREP, **held),
        blocked_seconds=300,
        blocked_release_seconds=release,
    )


def test_the_ready_recipes_are_released_early_and_pass(tmp_path, gateway) -> None:
    sweep, _, _ = _make_stalled(tmp_path, gateway, release=60)
    sweep.run()
    assert _status(sweep, "bad")["failure_class"] == "admission-stalled"
    # In the profile the blocked recipe was not loaded again, and the others ran in one load.
    assert "released" in _events(sweep)
    assert _status(sweep, "good1")["status"] == "passed"
    assert _status(sweep, "good2")["status"] == "passed"


def test_a_blocker_that_names_nobody_requeues_all_without_blame(
    tmp_path, gateway
) -> None:
    sweep, _, _ = _make_stalled(tmp_path, gateway, release=10**9, blocker_unnamed=True)
    for _ in range(2000):
        sweep.tick()
        sweep.clock.sleep(sweep.cfg.poll_seconds)
        if "no blocker names a recipe" in _events(sweep):
            break
    assert "no blocker names a recipe" in _events(sweep)
    assert "failed" not in _results(sweep)


def test_an_unnamed_blocker_is_deferred_after_bounded_observation(
    tmp_path, gateway
) -> None:
    sweep, _, _ = _make_stalled(tmp_path, gateway, release=10**9, blocker_unnamed=True)
    sweep.run()
    # Unknown attribution is deferred after bounded observation.
    assert _status(sweep, "bad")["status"] == "deferred"


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
