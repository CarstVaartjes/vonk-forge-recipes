"""A sweep started before the Controller serves its source waits for it, then tests it.

Recorded 2026-11-08: recipes main refreshed three proven recipes in place. A run
started after the publish workflow succeeded but before the Controller imported the
release read the old catalog, found every recipe Working and exited 0 having tested
nothing.
"""

from __future__ import annotations

import json
from dataclasses import replace
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


def test_a_source_the_catalog_never_serves_is_deferred_after_the_bound_not_tested(
    tmp_path: Path, gateway: Gateway
) -> None:
    """Skew is the sweep's own staleness: no job, no verdict, no fail evidence.

    Recorded 2026-11-19: the catalog moved to a revision whose slots the sweep's own
    qualification spec did not bind; the 409 was recorded as a recipe-data failure.
    """
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("good", ("1" * 64,), content=OLD)],
        [FakeModel("1" * 64)],
        gateway=gateway,
        authored={KEY: NEW},
    )
    _guard(clock, clock.now())
    assert sweep.run() == 0

    assert not [c for _, c in fleet.calls if c[:2] == ("profile", "add")]
    assert sweep.state.recipes[KEY]["status"] == "pending"
    assert "failure_class" not in sweep.state.recipes[KEY]
    rows = [r for r in _rows(tmp_path) if r["step"] == "catalog" and "recipe" not in r]
    assert [r["status"] for r in rows] == ["waiting", "wait-expired"]
    assert rows[1]["code"] == "sweep.wait_budget_exhausted"
    assert rows[1]["unserved"] == [{"recipe": KEY, "served": OLD, "authored": NEW}]
    assert "CATALOG DOES NOT SERVE THE SOURCE" in sweep.source_alert
    assert not sweep.pending()


def _adds(fleet, key: str) -> int:
    return len([c for _, c in fleet.calls if c[:3] == ("profile", "add", key)])


def test_a_revision_the_catalog_changes_mid_pass_is_deferred_by_the_one_gate(
    tmp_path: Path, gateway: Gateway
) -> None:
    """The incident through its production trigger: rows apply page by page.

    The fake catalog pages (two rows a page); the recipe changes under a sweep that
    already served its source. It must not be scheduled with the sweep's own spec
    (not even before the pass completes), and is tested once the source is served.
    """
    keys = [f"vonk-forge/{slug}" for slug in ("a", "b", "c")]
    recipes = [
        FakeRecipe(s, (str(i) * 64,), content=OLD) for i, s in enumerate("abc", 1)
    ]
    models = [FakeModel(str(i) * 64) for i in (1, 2, 3)]
    first, fleet, _ = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        authored={key: OLD for key in keys},
        restore_owner=2,
    )
    assert first.run() == 0 and first.source_serving is SourceServing.SERVED
    changed = keys[1]
    fleet.recipes[changed].content = NEW  # the library published a newer revision
    adds = _adds(fleet, changed)
    first.state.recipes[changed] = {"status": "pending"}  # due a test again
    loads = len([c for _, c in fleet.calls if c[:2] == ("profile", "load")])
    # A recipe merged since is served and authored: this run still has work to do.
    fleet.recipes["vonk-forge/d"] = FakeRecipe("d", ("4" * 64,), content=OLD)
    fleet.models["4" * 64] = FakeModel("4" * 64)
    first.authored["vonk-forge/d"] = OLD

    # Read the pages the way the sweep does, one a tick; the scheduler's view in
    # between (mid-pass, the pass not complete) must never include the recipe.
    read_new = held_mid_pass = False
    for _ in range(12):
        first.clock.sleep(first.cfg.catalog_seconds)
        first.advance_catalog(first.clock.now())
        if first.recipe_list.in_pass and first.recipes[changed].content_sha256 == NEW:
            read_new = True  # the new revision was read, the pass is not complete
            held_mid_pass = changed not in [r.key for r in first.pending()]
            assert held_mid_pass, "the changed row is pending mid-pass"
        if (
            not first.recipe_list.in_pass
            and first.recipes[changed].content_sha256 == NEW
        ):
            break
    assert read_new, "the changed row was never read mid-pass"
    assert not [r for r in first.pending() if r.key == changed]

    _guard(first.clock, first.clock.now())
    assert first.run() == 0  # a run with deferred recipes still ends
    assert _adds(fleet, changed) == adds
    assert first.state.recipes[changed].get("failure_class") is None
    owner = [(p, c) for p, c in fleet.calls if c[:2] == ("profile", "load")]
    assert _adds(fleet, "vonk-forge/d") > 0  # the served recipe was tested
    assert len(owner) > loads and owner[-1][0] == 2  # restored in this run

    served, _, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        fleet=fleet,
        gateway=gateway,
        authored={**{key: OLD for key in keys}, changed: NEW},
        restore_owner=2,
    )
    _guard(clock, clock.now())
    assert served.run() == 0  # the source is served: the recipe is tested now
    assert _adds(fleet, changed) > adds


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


def test_a_restart_continues_the_source_wait_budget(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("good", ("1" * 64,), content=OLD)]
    models = [FakeModel("1" * 64)]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, models, gateway=gateway, authored={KEY: NEW}
    )
    sweep.preflight()
    for _ in range(5):
        sweep.tick()
        sweep.observe_source_wait()
        clock.sleep(10)
    assert sweep.source_serving is SourceServing.UNSERVED
    started = sweep.source_watch["wait_started_at"]
    sweep.state.save()

    again, _, _ = make_sweep(
        tmp_path, recipes, models, fleet=fleet, authored={KEY: NEW}
    )
    assert again.source_watch["wait_started_at"] == started


def test_only_that_matches_no_recipe_is_an_error_not_a_clean_exit(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("good", ("1" * 64,), content=OLD)]
    sweep, _, _ = make_sweep(
        tmp_path, recipes, [FakeModel("1" * 64)], gateway=gateway, only=("typo",)
    )
    with pytest.raises(RuntimeError, match="--only typo matches no recipe"):
        sweep.run()


def test_a_restart_with_other_authored_content_waits_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("good", ("1" * 64,), content=OLD)]
    models = [FakeModel("1" * 64)]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, models, gateway=gateway, authored={KEY: NEW}
    )
    sweep.preflight()
    for _ in range(5):
        sweep.tick()
        sweep.observe_source_wait()
        clock.sleep(10)
    assert "wait_started_at" in sweep.source_watch
    sweep.state.save()

    newer = "e" * 64  # the source checkout moved on while the sweep was down
    again, _, _ = make_sweep(
        tmp_path, recipes, models, fleet=fleet, authored={KEY: newer}
    )
    again.preflight()
    again.tick()
    assert again.source_serving is SourceServing.UNSERVED
    assert again.source_watch.get("wait_started_at") != sweep.source_watch.get(
        "wait_started_at"
    )


def test_watch_mode_keeps_polling_when_only_matches_nothing_yet(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("good", ("1" * 64,), content=OLD)]
    sweep, _, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("1" * 64)],
        gateway=gateway,
        only=("not-yet-published",),
        watch_seconds=60.0,
    )
    start = clock.now()

    def stop_later(now: float) -> None:
        if now - start >= 600:
            raise KeyboardInterrupt

    clock.hooks.append(stop_later)
    assert sweep.run() == 130  # interrupted while still watching, no --only error


def test_a_recipe_the_older_source_lacks_is_held_per_row_on_any_pass(
    tmp_path: Path, gateway: Gateway
) -> None:
    newer = "vonk-forge/newer"
    sweep, _, _ = make_sweep(
        tmp_path,
        [FakeRecipe("good", ("1" * 64,), content=OLD)],
        [FakeModel("1" * 64)],
        gateway=gateway,
    )
    assert sweep.run() == 0 and sweep.catalog_loaded
    # A later pass lists a recipe merged since this sweep's source was taken.
    sweep.recipes[newer] = replace(sweep.recipes[KEY], key=newer)
    sweep.state.recipes[newer] = {"status": "pending"}
    assert [r.key for r in sweep.pending()] == []
    assert not sweep._open(newer)
    authored = {**sweep.authored, newer: OLD}
    sweep.authored = authored  # a restart from a newer source authors it
    assert [r.key for r in sweep.pending()] == [newer]
