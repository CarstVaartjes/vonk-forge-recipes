"""An ended Controller wait receipt is planned again later, never followed astray.

Behaviour through ``_tick()`` against the faithful fake: a pending admission ends
superseded by the newer load that takes its Sparks (``superseded_by``), or with no
successor once a newer profile save reaches its next re-plan.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.lifecycle import LifecycleState, RecipeStatus
from spark_sweep.policy import TimeoutPolicy
from spark_sweep.report import build_status, render_status

DIGEST = "1" * 64
WAIT = 600.0  # every lane's local wait budget
REPLAN_CAP = 3600.0  # run.ENDED_RECEIPT_REPLAN_CAP


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _recipe(slug: str, **kwargs: Any) -> FakeRecipe:
    return FakeRecipe(slug, (DIGEST,), content=slug[0] * 64, local="cached", **kwargs)


def _make(tmp_path, gateway, recipes):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel(DIGEST, local="cached")],
        gateway=gateway,
        timeouts=TimeoutPolicy(
            first_start_seconds=WAIT, floor_seconds=WAIT, cap_seconds=WAIT
        ),
    )
    sweep.preflight()
    sweep.refresh_catalog()
    return sweep, fleet, clock


def _placed(fleet, key: str) -> list[float]:
    """When the sweep put ``key`` into the sweep profile (one entry per placement)."""
    return sorted(
        {
            at
            for (profile, args), at in zip(fleet.calls, fleet.call_times)
            if profile == 10 and args[:3] == ("profile", "add", key)
        }
    )


def _one_dependency(app: dict[str, Any]) -> dict[str, Any]:
    """A bound child's typed dependency (child waits of two applications can share it).

    Never a pending admission's wait: the Controller stamps that with the application
    itself (the fake's default ``_admission_wait``).
    """
    reason = "observation-unavailable"
    return {
        "reason": reason,
        "blockers": [
            {
                "code": reason,
                "detail": reason,
                "operation": {"id": "dependency-1", "kind": "fleet-profile"},
                "severity": "warning",
            }
        ],
    }


def _passed(sweep, recipe: FakeRecipe) -> bool:
    return sweep.state.entry(recipe.key).get("status") == RecipeStatus.PASSED


def test_a_receipt_that_keeps_ending_is_planned_again_at_growing_intervals(
    tmp_path, gateway
) -> None:
    held = _recipe("a", blocked=True)
    others = [_recipe(f"{slug}x", load_seconds=500) for slug in "bcd"]
    sweep, fleet, clock = _make(tmp_path, gateway, [held, *others])
    start = clock.now()
    shown = False
    while clock.now() - start < 12 * 3600:
        sweep._tick()
        hold = sweep.controller_wait_holds().get(held.key)
        if hold is not None and hold.model_dump().get("not_before") and not shown:
            # The re-plan is named in status, with its time.
            assert "receipt ended; planned again at" in render_status(
                build_status(sweep)
            )
            shown = True
        clock.sleep(10)
    # The other Spark keeps getting work: every other recipe passed.
    assert all(_passed(sweep, r) for r in others)
    placed = _placed(fleet, held.key)
    gaps = [later - earlier for earlier, later in pairwise(placed)]
    assert len(gaps) >= 4
    # Never back to back: each re-placement waits at least as long as the one
    # before (time-based doubling) until the ceiling, then stays at the ceiling.
    assert all(gap > WAIT + sweep.cfg.defer_delay for gap in gaps)
    assert all(later >= min(earlier, REPLAN_CAP) for earlier, later in pairwise(gaps))
    assert gaps[-1] >= REPLAN_CAP
    assert sweep.state.entry(held.key).get("attempts", 0) == 0
    assert shown


def test_successor_of_another_recipe_is_not_followed_or_shared(
    tmp_path, gateway, monkeypatch
) -> None:
    # A waits for admission on spark-a for ever. Once its lane is released, B takes
    # spark-a: B's admission retires A's receipt (superseded_by = B's load), and B's
    # bound child then waits on the same typed dependency: the Controller still works
    # that receipt, so B keeps its lane while the other Spark keeps taking work.
    held = _recipe("a", blocked=True)
    taker = _recipe("b", load_seconds=10**6)
    later = [_recipe(f"{slug}x", load_seconds=300) for slug in "cdef"]
    sweep, fleet, clock = _make(tmp_path, gateway, [held, taker, *later])
    shown = fleet._app_doc

    def document(app):
        doc = shown(app)
        if not app.get("admission_pending") and any(
            item["alias"] == "b" for item in app["assign"]
        ):
            doc["wait"] = _one_dependency(app)
        return doc

    monkeypatch.setattr(fleet, "_app_doc", document)
    first: str | None = None
    for _ in range(900):
        sweep._tick()
        hold = sweep.controller_wait_holds().get(held.key)
        if first is None and hold is not None:
            first = hold.application_id
        # Another recipe's load is no continuation of A's receipt, so it cannot
        # make B's wait look shared: nothing pauses the fleet.
        assert sweep.controller_wait_issue() is None
        if all(_passed(sweep, r) for r in later):
            break
        clock.sleep(10)
    assert first is not None
    holds = sweep.controller_wait_holds()
    slot = sweep.state.slots[taker.key]
    assert slot["phase"] == "loading" and taker.key not in holds
    assert fleet.apps[first]["state"] == LifecycleState.SUPERSEDED
    assert (
        fleet.apps[first]["superseded_by"]
        == sweep.state.data["loads"][slot["request_key"]]["app_id"]
    )
    # A's hold never followed into B's load: it ended and plans A again later.
    assert holds[held.key].application_id == first
    assert held.key not in {r.key for r in sweep.ready_candidates()}
    # Unrelated recipes kept being placed on the other Spark past B's local window.
    assert all(_passed(sweep, r) for r in later)
    assert max(_placed(fleet, later[-1].key)) > min(_placed(fleet, taker.key)) + WAIT


def test_two_expired_pending_admissions_replan_and_admit_fresh_requests(
    tmp_path, gateway
) -> None:
    # A and B each wait for admission on their own application (the Controller's
    # defer path: persisted_waits.schedule_wait stamps the application itself), so
    # their waits are never shared. Each receipt ends superseded with no successor
    # (the sweep's own saves retired it at its next re-plan) and each recipe gets a
    # fresh request that is admitted once the Controller's hold ends.
    a = _recipe("a", blocked=True, blocked_for=3600)
    b = _recipe("b", blocked=True, blocked_for=3600)
    sweep, fleet, clock = _make(tmp_path, gateway, [a, b])
    start = clock.now()
    receipts: dict[str, str] = {}
    for _ in range(900):
        sweep._tick()
        for recipe in (a, b):
            hold = sweep.controller_wait_holds().get(recipe.key)
            if hold is not None:
                receipts.setdefault(recipe.key, hold.application_id)
        # Two admission waits, each on its own application: never one incident.
        assert sweep.controller_wait_issue() is None
        if _passed(sweep, a) and _passed(sweep, b):
            break
        clock.sleep(10)
    assert set(receipts) == {a.key, b.key}
    assert receipts[a.key] != receipts[b.key]
    ended = [fleet.apps[app] for app in receipts.values()]
    assert {app["state"] for app in ended} == {LifecycleState.SUPERSEDED}
    assert all(app.get("superseded_by") is None for app in ended)
    # Each recipe got a fresh request after its first receipt and passed.
    for recipe in (a, b):
        first = fleet.apps[receipts[recipe.key]]["created"]
        assert max(_placed(fleet, recipe.key)) > first
        assert _passed(sweep, recipe)
    assert clock.now() - start < 3600 + 2 * REPLAN_CAP
