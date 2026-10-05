"""A load the Controller replaced is followed or requeued, never recorded as a recipe failure."""

from __future__ import annotations

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy

SID = "8cb1d3a7-0000-4000-8000-000000000001"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


@pytest.mark.parametrize("mode", ["retry", "legacy-retry"])
def test_an_automatic_retry_is_adopted_and_the_recipe_passes(
    tmp_path, gateway, mode
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.supersede = [mode]
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


def test_supersession_is_read_from_state_code_and_legacy_text() -> None:
    first = policy.supersession({"state": "superseded", "superseded_by": SID})
    assert first is not None and first.successor == SID
    legacy = {
        "state": "failed",
        "status_reason": f"Automatically reconciled by profile retry {SID}",
    }
    found = policy.supersession(legacy)
    assert found is not None and found.successor == SID
    cancelled = policy.supersession(
        {
            "state": "cancelled",
            "status_reason": "Pending profile intent was superseded: Profile workload effects changed during admission; review again",
        }
    )
    assert cancelled is not None and cancelled.successor is None
    assert policy.supersession({"state": "failed", "status_reason": "boom"}) is None
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
    sweep.state.save()
    again, _, _ = make_sweep(tmp_path, recipes, models, fleet=fleet)
    again.run()
    assert _entry(again, "a")["status"] == "passed"
    assert _entry(again, "b")["status"] == "failed"
