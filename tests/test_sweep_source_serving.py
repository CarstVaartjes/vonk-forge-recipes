"""A sweep started before the Controller serves its source waits for it, then tests it.

Recorded 2026-11-08: recipes main refreshed three proven recipes in place. A run
started after the publish workflow succeeded but before the Controller imported the
release read the old catalog, found every recipe Working and exited 0 having tested
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.source_serving import SourceServing, Unserved, advance, unserved

KEY = "vonk-forge/good"
OLD = "c" * 64
NEW = "d" * 64


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _rows(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]


def test_a_refreshed_proven_recipe_waits_for_the_catalog_then_is_retested(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("good", ("1" * 64,), content=OLD)]
    models = [FakeModel("1" * 64)]
    first, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    assert first.run() == 0
    assert first.state.recipes[KEY]["status"] == "passed"
    adds_before = len([c for p, c in fleet.calls if c[:2] == ("profile", "add")])

    # The source refreshed the recipe in place; the Controller still serves the old content.
    sweep, _, clock = make_sweep(
        tmp_path, recipes, models, fleet=fleet, authored={KEY: NEW}
    )
    start = clock.now()
    before_import: list[tuple[bool, int, tuple[Unserved, ...]]] = []

    def import_release(now: float) -> None:
        if now - start >= 600 and not before_import:
            adds = len([c for p, c in fleet.calls if c[:2] == ("profile", "add")])
            before_import.append((sweep.done(), adds, sweep.unserved_recipes))
            fleet.recipes[KEY].content = NEW
        if now - start >= 20_000:
            raise AssertionError("the sweep never finished after the import")

    clock.hooks.append(import_release)
    assert sweep.run() == 0

    # Before the import: not done, nothing loaded, and the wait names the recipe.
    assert before_import, "the run ended before the catalog served its source"
    done, adds, waiting = before_import[0]
    assert not done and adds == adds_before
    assert waiting == (Unserved(KEY, OLD, NEW),)
    # After the import: the proven recipe moved, so it is re-checked on the new content.
    entry = sweep.state.recipes[KEY]
    assert entry["status"] == "passed" and entry["content_sha256"] == NEW
    rows = _rows(tmp_path)
    catalog = [
        r["status"] for r in rows if r["step"] == "catalog" and "recipe" not in r
    ]
    assert catalog == ["waiting", "available"]
    requeue = [r for r in rows if r.get("recipe") == KEY and r["step"] == "requeue"]
    assert requeue and requeue[-1]["reason"] == "passed-on-older-revision"
    assert sweep.source_serving is SourceServing.SERVED


def test_a_served_source_is_not_a_ceiling_and_absence_is_not_judged() -> None:
    differing = (Unserved(KEY, OLD, NEW),)
    # Only a complete pass decides.
    assert (
        advance(SourceServing.UNREAD, complete_pass=False, differing=differing)
        is SourceServing.UNREAD
    )
    assert (
        advance(SourceServing.UNREAD, complete_pass=True, differing=differing)
        is SourceServing.UNSERVED
    )
    assert (
        advance(SourceServing.UNSERVED, complete_pass=True, differing=())
        is SourceServing.SERVED
    )
    # Once served, a newer library (a merge during a long sweep) is followed, not awaited.
    assert (
        advance(SourceServing.SERVED, complete_pass=True, differing=differing)
        is SourceServing.SERVED
    )
    # A recipe on one side only, or outside --only, is not a difference.
    assert unserved(
        {KEY: OLD, "vonk-forge/listed-only": OLD, "vonk-forge/other": OLD},
        {KEY: NEW, "vonk-forge/source-only": NEW, "vonk-forge/other": NEW},
        lambda key: key != "vonk-forge/other",
    ) == (Unserved(KEY, OLD, NEW),)
