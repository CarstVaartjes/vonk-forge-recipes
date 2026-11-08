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

from spark_sweep.source_serving import (
    SourceEvent,
    SourceServing,
    Unserved,
    advance,
    unserved,
)

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
    assert catalog == ["waiting", "served"]
    requeue = [r for r in rows if r.get("recipe") == KEY and r["step"] == "requeue"]
    assert requeue and requeue[-1]["reason"] == "passed-on-older-revision"
    assert sweep.source_serving is SourceServing.SERVED


def _first_load_at(fleet) -> float:
    return next(
        t
        for t, (_, c) in zip(fleet.call_times, fleet.calls)
        if c[:2] == ("profile", "add")
    )


def _guard(clock, start: float) -> None:
    def hang(now: float) -> None:
        if now - start >= 20_000:
            raise AssertionError("the sweep never finished")

    clock.hooks.append(hang)


def test_a_source_the_catalog_never_serves_is_waited_for_within_the_bound_only(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("good", ("1" * 64,), content=OLD)],
        [FakeModel("1" * 64)],
        gateway=gateway,
        authored={KEY: NEW},
        restore_owner=2,
    )
    start = clock.now()
    _guard(clock, start)
    assert sweep.run() == 0

    # Nothing loaded within the bound; after it, the served content was tested.
    assert _first_load_at(fleet) >= start + sweep.cfg.source_wait_seconds
    entry = sweep.state.recipes[KEY]
    assert entry["status"] == "passed" and entry["content_sha256"] == OLD
    rows = [r for r in _rows(tmp_path) if r["step"] == "catalog" and "recipe" not in r]
    assert [r["status"] for r in rows] == ["waiting", "wait-expired"]
    expired = rows[-1]
    assert expired["code"] == "sweep.wait_budget_exhausted"
    assert expired["unserved"] == [{"recipe": KEY, "served": OLD, "authored": NEW}]
    assert expired["wait_expiry"]["wait"]["reason"] == "catalog source"
    assert "CATALOG DOES NOT SERVE THE SOURCE" in sweep.source_alert
    # The run finished as usual: the owner's profile was loaded back last.
    last = [(p, c) for p, c in fleet.calls if c[:2] == ("profile", "load")][-1]
    assert last[0] == 2


def test_a_new_library_that_leaves_the_recipe_unchanged_keeps_it_waiting(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("good", ("1" * 64,), content=OLD)],
        [FakeModel("1" * 64)],
        gateway=gateway,
        authored={KEY: NEW},
    )
    start = clock.now()
    seen: list[tuple[bool, int]] = []

    def releases(now: float) -> None:
        if now - start >= 120 and fleet.library_commit == "lib-commit-1":
            fleet.library_commit = "lib-commit-2"  # an unrelated release: same digest
        if now - start >= 600 and not seen:
            adds = len([c for _, c in fleet.calls if c[:2] == ("profile", "add")])
            seen.append((sweep.done(), adds))
            fleet.recipes[KEY].content = NEW  # the release with the source arrives
            fleet.library_commit = "lib-commit-3"

    clock.hooks.append(releases)
    _guard(clock, start)
    assert sweep.run() == 0

    assert seen == [(False, 0)]  # still waiting across the unrelated release
    entry = sweep.state.recipes[KEY]
    assert entry["status"] == "passed" and entry["content_sha256"] == NEW
    rows = [r for r in _rows(tmp_path) if r["step"] == "catalog" and "recipe" not in r]
    assert [r["status"] for r in rows] == ["waiting", "served"]
    assert not sweep.source_alert


def test_the_source_state_machine_and_what_is_judged() -> None:
    S, E = SourceServing, SourceEvent
    assert advance(S.UNREAD, E.UNSERVED_PASS) is S.UNSERVED
    assert advance(S.UNREAD, E.SERVED_PASS) is S.SERVED
    assert advance(S.UNSERVED, E.UNSERVED_PASS) is S.UNSERVED
    for ending in (E.SERVED_PASS, E.WAIT_EXPIRED):
        assert advance(S.UNSERVED, ending) is S.SERVED
    # Once following the catalog, newer library content is tested, never awaited.
    for event in E:
        assert advance(S.SERVED, event) is S.SERVED
    # A recipe on one side only, or outside --only, is not a difference.
    assert unserved(
        {KEY: OLD, "vonk-forge/listed-only": OLD, "vonk-forge/other": OLD},
        {KEY: NEW, "vonk-forge/source-only": NEW, "vonk-forge/other": NEW},
        lambda key: key != "vonk-forge/other",
    ) == (Unserved(KEY, OLD, NEW),)
