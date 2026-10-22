"""Unanswered canonical reviews re-plan without a recipe verdict or idle lanes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts.sweep import (
    CliObservation,
    CliObservationReply,
    CliObservationStatus,
    InfrastructureIncident,
)

from spark_sweep.lifecycle import RecipeStatus, RetestReason
from spark_sweep.report import build_status, render_status

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

OBSERVATION = CliObservationReply(
    observation=CliObservation(
        status=CliObservationStatus.TIMED_OUT,
        error="control API connection was lost",
    ),
    result={},
).model_dump(mode="json")
TRANSPORT = json.loads(
    (
        Path(__file__).parent / "fixtures/sweep-verdicts/cli-transport-timeout.json"
    ).read_text()
)


@pytest.mark.parametrize("reply", [OBSERVATION, TRANSPORT, None])
def test_dual_review_outage_fills_both_single_lanes_and_recovers_after_restart(
    tmp_path: Path, reply: object
) -> None:
    duals = [
        FakeRecipe(f"dual-{i}", node_count=2, local="cached")
        for i in reversed(range(2))
    ]
    singles = [FakeRecipe(f"single-{i}", local="cached") for i in range(2)]
    recipes = duals + singles
    models = [FakeModel("m1", local="cached")]
    gateway = Gateway()
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, models, gateway=gateway, boost=tuple(r.slug for r in duals)
    )
    # Reproduce the refreshed platform-blocked priority that put duals first.
    for recipe in duals:
        sweep.state.entry(recipe.key)["requeued_because"] = (
            RetestReason.DEFERRED_OLDER_REVISION
        )
    original = fleet.runner
    failed: list[str] = []
    recovering = False

    def preview_outage(argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        assignments = fleet.profiles.get(sweep.cfg.sweep_profile, {}).get(
            "assignments", []
        )
        if (
            not recovering
            and "--review" in argv
            and any(
                fleet.recipes[a["recipe_selector"]].node_count == 2 for a in assignments
            )
        ):
            failed.extend(a["recipe_selector"] for a in assignments)
            clock.sleep(30)  # real preview's bounded observation timeout
            return 2, json.dumps(reply), "control API connection was lost"
        return original(argv, timeout)

    sweep.vk.runner = preview_outage
    try:
        sweep.preflight()
        sweep.tick()
        assert failed == [duals[0].key]  # one probe per failed topology
        assert set(sweep.state.slots) == {r.key for r in singles}
        assert {tuple(s["node_ids"]) for s in sweep.state.slots.values()} == {
            ("spk_a",),
            ("spk_b",),
        }
        for recipe in duals:
            assert sweep.state.status(recipe.key) == RecipeStatus.PENDING
            assert sweep.state.entry(recipe.key).get("attempts", 0) == 0
        page = render_status(build_status(sweep))
        assert "Infrastructure problems (not recipe failures; retrying)" in page
        assert "control API connection was lost" in page or "timed out" in page
        assert not (tmp_path / "results.jsonl").exists()
        sweep.state.save()
        restarted, _, _ = make_sweep(
            tmp_path, recipes, models, fleet=fleet, boost=tuple(r.slug for r in duals)
        )
        restarted.vk.runner = preview_outage
        # Singles remain owned; no re-review can replace their accepted effects.
        before = dict(sweep.state.slots)
        restarted.tick()
        assert set(restarted.state.slots) == set(before)
        assert failed == [duals[0].key]
        recovering = True
        assert restarted.run() == 0
        assert all(
            restarted.state.status(r.key) == RecipeStatus.PASSED for r in recipes
        )
        assert not restarted.state.data["infra"]
        assert "review.blocked" not in (tmp_path / "results.jsonl").read_text()
    finally:
        gateway.stop()


def test_review_condition_backoff_grows_is_durable_and_does_not_blame_recipe(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("dual", node_count=2, local="cached")
    models = [FakeModel("m1", local="cached")]
    sweep, fleet, clock = make_sweep(tmp_path, [recipe], models, retry_delay=300)
    for attempt in range(1, 4):
        fleet.observations.append((("profile", "load", "--review"), 2, OBSERVATION))
        sweep.tick()
        source = sweep._review_transport_source(sweep.recipes[recipe.key])
        incident = InfrastructureIncident.model_validate(
            sweep.state.data["infra"][source]
        )
        assert incident.count == attempt
        assert incident.until - clock.now() == 30 * 2 ** (attempt - 1)
        assert sweep.state.status(recipe.key) == RecipeStatus.PENDING
        assert sweep.state.entry(recipe.key).get("attempts", 0) == 0
        sweep.state.save()
        sweep, _, _ = make_sweep(
            tmp_path, [recipe], models, fleet=fleet, retry_delay=300
        )
        sweep.tick()  # restart and polling cannot renew or bypass the deadline
        assert sweep.state.data["infra"][source]["count"] == attempt
        assert not sweep.state.slots
        clock.sleep(incident.until - clock.now())
    # A fresh allowed canonical review clears exactly this incident.
    sweep.tick()
    assert recipe.key in sweep.state.slots
    assert source not in sweep.state.data["infra"]
    assert (
        not (tmp_path / "results.jsonl").exists()
        or "review.blocked" not in (tmp_path / "results.jsonl").read_text()
    )
