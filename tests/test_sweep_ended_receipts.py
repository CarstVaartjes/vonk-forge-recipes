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
    """Waits on one typed dependency (the same operation for every application)."""
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
    # bound child then waits on the same typed dependency until its own budget ends.
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
    assert fleet.apps[first]["state"] == LifecycleState.SUPERSEDED
    assert fleet.apps[first]["superseded_by"] == holds[taker.key].application_id
    # A's hold never followed into B's load: it ended and plans A again later.
    assert holds[held.key].application_id == first
    assert held.key not in {r.key for r in sweep.ready_candidates()}
    # Unrelated recipes kept being placed after B's wait expired.
    assert all(_passed(sweep, r) for r in later)
    assert max(_placed(fleet, later[-1].key)) > min(_placed(fleet, taker.key)) + WAIT


def test_two_expired_pending_admissions_heal_and_admit_fresh_work(
    tmp_path, gateway, monkeypatch
) -> None:
    # A and B wait for admission on one dependency. B's expiry lands while A's
    # receipt cannot be read, so both holds are live: a shared incident pauses
    # placement. The partition heals and both receipts are found ended (newer saves
    # retired them at their re-plan, with no successor).
    a = _recipe("a", blocked=True, blocked_for=3600)
    b = _recipe("b", blocked=True, blocked_for=3600)
    fresh = _recipe("c")
    sweep, fleet, clock = _make(tmp_path, gateway, [a, b, fresh])
    monkeypatch.setattr(fleet, "_admission_wait", _one_dependency)
    read = fleet._app_progress
    unreadable: set[str] = set()

    def progress(data, args):
        if "--application" in args and args[args.index("--application") + 1] in (
            unreadable
        ):
            return 1, None  # a partitioned read
        return read(data, args)

    monkeypatch.setattr(fleet, "_app_progress", progress)
    start = clock.now()
    shared_at: float | None = None
    for _ in range(900):
        sweep._tick()
        holds = sweep.controller_wait_holds()
        if shared_at is None and a.key in holds:
            unreadable.add(holds[a.key].application_id)
        if shared_at is None and sweep.controller_wait_issue() is not None:
            shared_at = clock.now()
            assert not sweep.ready_candidates()
            unreadable.clear()  # the partition heals
        if _passed(sweep, fresh):
            break
        clock.sleep(10)
    assert shared_at is not None, "the scenario must reach a shared incident"
    ended = [fleet.apps[hold.application_id] for hold in holds.values()]
    assert {app["state"] for app in ended} == {LifecycleState.SUPERSEDED}
    assert all(app.get("superseded_by") is None for app in ended)
    # Ended receipts prove nothing any more: fresh work is admitted promptly.
    (placed,) = _placed(fleet, fresh.key)
    assert placed - shared_at <= 3 * sweep.cfg.defer_delay
    assert _passed(sweep, fresh)
    assert sweep.controller_wait_issue() is None
    # The held recipes are planned again later and pass once admitted.
    for _ in range(900):
        if _passed(sweep, a) and _passed(sweep, b):
            break
        clock.sleep(10)
        sweep._tick()
    assert _passed(sweep, a) and _passed(sweep, b)
    assert clock.now() - start < 6 * 3600
