"""A cause the Controller or the fleet can still change is followed or retried, not a recipe failure."""

from __future__ import annotations

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep import verdict
from spark_sweep.identity import content_identity
from spark_sweep.run import MAX_REVIEW_FREEING


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _refusing(fleet, times: int | None):
    """Review refuses on capacity ``times`` times (always when None), then reviews as usual."""

    real = fleet._load
    refused = []

    def load(n, data, a):
        if "--review" in a and (times is None or len(refused) < times):
            refused.append(1)
            return 2, {
                "allowed": False,
                **fleet.review_assignments(data),
                "effects": {"runs": []},
                "admission_decisions": [
                    {
                        "alias": "a",
                        "allowed": False,
                        "blockers": [
                            {
                                "code": "run-switch.resource.insufficient_capacity",
                                "detail": "7603704832 bytes short of the reserve",
                            }
                        ],
                    }
                ],
            }
        return real(n, data, a)

    fleet._load = load
    return refused


def test_a_capacity_refusal_that_clearing_undoes_is_requeued_not_failed(
    tmp_path, gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    refused = _refusing(fleet, times=1)
    assert sweep.run() == 0
    assert len(refused) == 1
    assert _entry(sweep, "a")["status"] == "passed"
    assert "clearing the fleet and trying again" in str(sweep.state.data["events"])


def _waits(sweep, key: str) -> list[dict]:
    return [
        r
        for r in sweep.log.entries()
        if r.get("recipe") == key and r.get("disposition") == verdict.Disposition.WAIT
    ]


def test_a_capacity_refusal_that_outlasts_clearing_waits(tmp_path, gateway) -> None:
    # A Controller build (or any holder clearing cannot stop) keeps the room past the
    # clearings: it waits and is reviewed again (the Controller owns the fit).
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    refused = _refusing(fleet, times=MAX_REVIEW_FREEING + 3)
    assert sweep.run() == 0
    entry = _entry(sweep, "a")
    assert entry["status"] == "passed"
    assert len(refused) == MAX_REVIEW_FREEING + 3  # reviewed again after each wait
    assert len(_waits(sweep, "vonk-forge/a")) >= 3
    rows = [r for r in sweep.log.entries() if r.get("recipe") == "vonk-forge/a"]
    assert not [r for r in rows if r.get("status") in ("deferred", "failed")]


def test_a_recipe_declaring_more_than_an_idle_spark_is_placed_and_passes(
    tmp_path, gateway
) -> None:
    # The Controller tries one oversized estimate on a Spark that holds nothing.
    huge = FakeRecipe("huge", memory=128 * 10**9)
    sweep, _, _ = make_sweep(tmp_path, [huge], [FakeModel("m1")], gateway=gateway)
    assert sweep.run() == 0
    assert _entry(sweep, "huge")["status"] == "passed"


def test_a_stored_review_capacity_deferral_is_reopened_and_passes(
    tmp_path, gateway
) -> None:
    recipe = FakeRecipe("a")
    first, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    first.refresh_catalog()
    first.state.entry(recipe.key).update(
        status="deferred",
        phase="review",
        failure_class="capacity",
        code="run-switch.resource.insufficient_capacity",
        attempts=1,
        evidence={"demand_exceeds_idle_spark": True},
        verdict_identity=content_identity(first.recipes[recipe.key]).model_dump(
            mode="json"
        ),
    )
    first.state.save()
    resumed, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    assert resumed.run() == 0
    assert resumed.state.recipes[recipe.key]["status"] == "passed"


def test_a_failed_application_the_controller_gave_up_on_is_recorded(
    tmp_path, gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.supersede = ["failed-repeated"]
    sweep.run()
    entry = _entry(sweep, "a")
    assert (entry["status"], entry["phase"]) == ("deferred", "start")


@pytest.mark.parametrize("solo_refused", [False, True])
def test_one_refused_lane_does_not_starve_an_admitted_sibling(
    tmp_path, gateway, solo_refused
) -> None:
    """A build holds a's Spark; the rejected batch must be reviewed again with b alone."""
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [FakeRecipe("a", memory=100 * 10**9), FakeRecipe("b", memory=100 * 10**9)],
        [FakeModel("m1")],
        gateway=gateway,
    )
    real = fleet._load
    refused = _refusing(fleet, times=None)
    refusing = fleet._load
    admitted = []

    def load(n, data, a):
        aliases = {item["assignment_name"] for item in data["assignments"]}
        if "--review" in a and "a" in aliases and not admitted:
            status, document = refusing(n, data, a)
            if "b" in aliases:
                document["admission_decisions"].append({"alias": "b", "allowed": True})
            return status, document
        if "--review" in a and aliases == {"b"} and solo_refused:
            status, document = refusing(n, data, a)
            document["admission_decisions"][0]["alias"] = "b"
            return status, document
        if "--review" not in a and aliases == {"b"}:
            admitted.append(aliases)
        return real(n, data, a)

    fleet._load = load
    from spark_sweep.policy import Placement

    fleet.runs.clear()
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._apply(
        [
            Placement(sweep.recipes[f"vonk-forge/{slug}"], (node,))
            for slug, node in (("a", "spk_a"), ("b", "spk_b"))
        ],
        sweep.clock.now(),
    )
    assert bool(admitted) is not solo_refused
    assert len(refused) == (2 if solo_refused else 1)
    assert ("vonk-forge/b" in sweep.state.slots) is not solo_refused
    assert "vonk-forge/a" not in sweep.state.slots
    if not solo_refused:
        assert sweep.run() == 0
        assert all(_entry(sweep, slug)["status"] == "passed" for slug in ("a", "b"))
