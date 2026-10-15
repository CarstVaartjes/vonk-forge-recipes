"""A cause the Controller or the fleet can still change is followed or retried, not a recipe failure."""

from __future__ import annotations

import pytest
from sweep_fakes import SPARK_MEMORY, FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep import verdict
from spark_sweep.run import MAX_REVIEW_FREEING


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _refusing(fleet, times: int | None, demand: int | None = 40 * 10**9):
    """Review refuses on capacity ``times`` times (always when None), then reviews as usual.

    ``demand`` is the Controller's own memory requirement for the Spark (None: unknown).
    """

    real = fleet._load
    refused = []

    def load(n, data, a):
        if "--review" in a and (times is None or len(refused) < times):
            refused.append(1)
            return 2, {
                "allowed": False,
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
                        "requirements": [
                            {
                                "node_id": node,
                                "memory_required_bytes": demand,
                                "memory_floor_bytes": 6 * 10**9,
                            }
                            for node, *_ in (fleet.sparks if demand is not None else ())
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


@pytest.mark.parametrize("demand", [40 * 10**9, None], ids=["fits-idle", "unknown"])
def test_a_capacity_refusal_that_outlasts_clearing_waits_when_it_can_fit(
    tmp_path, gateway, demand
) -> None:
    # A Controller build (or any holder clearing cannot stop) keeps the room past the
    # clearings: the recipe fits an idle Spark (or that is unknown), so it waits.
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    refused = _refusing(fleet, times=MAX_REVIEW_FREEING + 3, demand=demand)
    assert sweep.run() == 0
    entry = _entry(sweep, "a")
    assert entry["status"] == "passed"
    assert len(refused) == MAX_REVIEW_FREEING + 3  # reviewed again after each wait
    assert len(_waits(sweep, "vonk-forge/a")) >= 3
    rows = [r for r in sweep.log.entries() if r.get("recipe") == "vonk-forge/a"]
    assert not [r for r in rows if r.get("status") in ("deferred", "failed")]


def test_a_capacity_refusal_beyond_an_idle_spark_is_deferred_with_evidence(
    tmp_path, gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    refused = _refusing(fleet, times=None, demand=SPARK_MEMORY)
    sweep.run()
    entry = _entry(sweep, "a")
    assert (entry["status"], entry["phase"], entry["failure_class"]) == (
        "deferred",
        "review",
        "capacity",
    )
    assert entry["evidence"]["freeing_attempts"] == MAX_REVIEW_FREEING
    assert len(refused) == MAX_REVIEW_FREEING + 1


def test_a_capacity_failure_recorded_before_clearing_existed_is_requeued_once(
    tmp_path, gateway
) -> None:
    sweep, _fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway, limit=0
    )
    entry = sweep.state.entry("vonk-forge/a")
    entry.update(
        status="failed",
        phase="review",
        failure_class="capacity",
        evidence={"code": "run-switch.resource.insufficient_capacity"},
    )
    sweep.requeue_unfreed_capacity_failures()
    assert entry["status"] == "pending"
    entry.update(
        status="failed",
        phase="review",
        failure_class="capacity",
        evidence={"freeing_attempts": 2},
    )
    sweep.requeue_unfreed_capacity_failures()
    assert entry["status"] == "failed"


@pytest.mark.parametrize("mode", ["failed-retry"])
def test_a_failed_application_the_controller_still_retries_is_followed(
    tmp_path, gateway, mode
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.supersede = [mode]
    assert sweep.run() == 0
    assert _entry(sweep, "a")["status"] == "passed"
    assert "the Controller retries it" in str(sweep.state.data["events"])


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


def test_the_controllers_typed_ending_is_what_stops_the_following() -> None:
    assert verdict.retry_ended(
        {"blockers": [{"code": "profile.failure_repeated", "detail": "x"}]}
    )
    assert verdict.retry_ended(
        {"progress": {"switch_adapter": {"assignment_failures": [{"terminal": True}]}}}
    )
    assert not verdict.retry_ended(
        {"progress": {"switch_adapter": {"assignment_failures": [{"terminal": False}]}}}
    )
