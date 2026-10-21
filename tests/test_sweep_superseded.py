"""A load the Controller replaced is followed or requeued, never recorded as a recipe failure."""

from __future__ import annotations

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy
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
def test_a_load_never_adopts_another_of_our_loads_as_its_continuation(
    tmp_path, gateway, observed
) -> None:
    sweep, _fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    load = _two_own_loads(sweep, observed)

    assert sweep._poll_application(load)

    assert load["app_id"] != NEW_APP or observed["id"] == NEW_APP
    assert "key-old" not in sweep.state.data["loads"]
    assert sweep.state.data["apps"]["key-old"]["state"] == "superseded"


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
