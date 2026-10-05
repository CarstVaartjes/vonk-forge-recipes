"""Fleet-dependent skips are derived and heal; the pin profile respects the contract limit."""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import prefetch
from spark_sweep.prefetch import PIN_PROFILE_MAX_ASSIGNMENTS, PrefetchConfig


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _status(sweep, slug: str) -> str:
    return sweep.state.recipes[f"vonk-forge/{slug}"]["status"]


def test_restart_with_a_spark_not_online_in_the_snapshot_keeps_duals_pending(
    tmp_path: Path, gateway: Gateway
) -> None:
    dual = FakeRecipe("dual", node_count=2)
    sweep, fleet, _ = make_sweep(
        tmp_path, [dual, FakeRecipe("one")], [FakeModel("m1")], gateway=gateway
    )
    fleet.flaky_fleet_reads = 6  # the restart's early reads see one Spark reconnecting
    sweep.run()
    assert _status(sweep, "dual") == "passed"
    assert _status(sweep, "one") == "passed"


def test_a_fleet_skip_is_not_persisted_as_a_verdict(
    tmp_path: Path, gateway: Gateway
) -> None:
    dual = FakeRecipe("dual", node_count=2)
    sweep, fleet, _ = make_sweep(tmp_path, [dual], [FakeModel("m1")], gateway=gateway)
    fleet.flaky_fleet_reads = 2  # preflight reads once, the first pass reads again
    sweep.preflight()
    sweep.refresh_catalog()  # the snapshot shows 1 online: skipped for now
    assert _status(sweep, "dual") == "skipped"
    sweep.refresh_catalog()  # the fleet is whole again: pending again
    assert _status(sweep, "dual") == "pending"
    assert "reason" not in sweep.state.recipes["vonk-forge/dual"]


def test_old_state_with_fleet_size_skips_heals_on_the_next_pass(
    tmp_path: Path, gateway: Gateway
) -> None:
    dual = FakeRecipe("dual", node_count=2)
    sweep, _, _ = make_sweep(tmp_path, [dual], [FakeModel("m1")], gateway=gateway)
    sweep.state.entry("vonk-forge/dual").update(
        status="skipped",
        reason="not testable on this fleet: needs 2 Sparks, 1 usable",
    )  # as an older sweep wrote it: no reason_kind
    sweep.run()
    assert _status(sweep, "dual") == "passed"


def test_a_skip_that_is_not_about_the_fleet_is_left_alone(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.state.entry("vonk-forge/a").update(status="skipped", reason="owner said so")
    sweep.refresh_catalog()
    sweep.heal_fleet_skips()
    assert _status(sweep, "a") == "skipped"


def test_more_pin_candidates_than_the_contract_allows_are_capped(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = 5  # the contract's 64, scaled down so the fake fleet reaches it
    monkeypatch.setattr(prefetch, "PIN_PROFILE_MAX_ASSIGNMENTS", limit)
    recipes = [FakeRecipe(f"r{i}", (f"m{i}",)) for i in range(14)]
    models = [FakeModel(f"m{i}") for i in range(14)]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        prefetch=PrefetchConfig(
            pin_profile=13, max_model_downloads=14, pins_per_tick=20
        ),
    )
    fleet.max_assignments = limit
    sizes: list[int] = []
    clock.hooks.append(
        lambda _t: sizes.append(len(fleet.profiles.get(13, {}).get("assignments", [])))
    )
    sweep.run()
    assert max(sizes) == limit  # filled to the limit, never past it
    assert sweep.prefetcher.pin_error is None
    assert sweep.prefetcher.pin_failures == 0
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}


def test_the_contract_limit_is_64() -> None:
    assert PIN_PROFILE_MAX_ASSIGNMENTS == 64


def test_the_pin_profile_is_counted_on_the_real_document_not_on_our_bookkeeping(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untracked leftovers (a timed-out add the Controller applied, a lost state) and foreign
    assignments count against the contract limit: the submitted document never exceeds it."""
    from spark_sweep.run import SWEEP_LABEL

    limit = 6
    monkeypatch.setattr(prefetch, "PIN_PROFILE_MAX_ASSIGNMENTS", limit)
    recipes = [FakeRecipe(f"r{i}", (f"m{i}",)) for i in range(10)]
    models = [FakeModel(f"m{i}") for i in range(10)]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        prefetch=PrefetchConfig(
            pin_profile=13, max_model_downloads=10, pins_per_tick=20
        ),
    )

    def assignment(name: str, key: str) -> dict:
        return {
            "recipe_selector": key,
            "spark_ids": ["spark-a"],
            "assignment_name": name,
            "desired_state": "installed",
        }

    fleet.max_assignments = limit
    fleet.profiles[13] = {
        "revision": 1,
        "latest": None,
        "labels": {SWEEP_LABEL[0]: SWEEP_LABEL[1]},
        "assignments": [assignment(f"keep-{i}", "vonk-forge/r0") for i in range(2)]
        + [assignment(f"pin-stale{i}", "vonk-forge/gone") for i in range(4)],
    }  # full at the limit, and none of it in the sweep's state
    sizes: list[int] = []
    clock.hooks.append(lambda _t: sizes.append(len(fleet.profiles[13]["assignments"])))
    sweep.run()
    assert max(sizes) <= limit
    assert sweep.prefetcher.pin_error is None
    names = {a["assignment_name"] for a in fleet.profiles[13]["assignments"]}
    assert {"keep-0", "keep-1"} <= names  # foreign assignments are never removed
    assert not any(n.startswith("pin-stale") for n in names)
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
