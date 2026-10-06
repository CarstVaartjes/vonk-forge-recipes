"""Class guards: published vocabulary, bounded waits and falling untyped access."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from spark_sweep.lifecycle import ACTIVE, STATES, TERMINAL, WAITING

ROOT = Path(__file__).resolve().parents[1]


def unbounded_whiles(source: str) -> list[int]:
    """A wait's deadline must be in its condition and must not advance in its body."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.While):
            continue
        deadline = any(
            isinstance(item, ast.Compare)
            and any(
                isinstance(n, ast.Name) and n.id == "deadline" for n in ast.walk(item)
            )
            for item in ast.walk(node.test)
        )
        extended = any(
            isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Store)
            and item.id == "deadline"
            for statement in node.body
            for item in ast.walk(statement)
        )
        if not deadline or extended:
            bad.append(node.lineno)
    return bad


def test_loop_guard_rejects_unbounded_and_moving_deadlines() -> None:
    assert unbounded_whiles("while True:\n    poll()") == [1]
    assert unbounded_whiles("while now() < deadline:\n    deadline += 1") == [1]
    assert (
        unbounded_whiles("deadline = now() + 60\nwhile now() < deadline:\n    poll()")
        == []
    )


def test_every_sweep_while_has_a_fixed_deadline() -> None:
    paths = [*sorted((ROOT / "spark_sweep").glob("*.py")), ROOT / "tools/sweep-recipes"]
    for path in paths:
        assert unbounded_whiles(path.read_text()) == [], str(path)


def test_vocabulary_matches_published_controller_contract() -> None:
    published = json.loads((ROOT / "tests/fixtures/sweep-lifecycle.json").read_text())
    assert STATES == frozenset(published["LifecycleState"]["enum"])
    assert ACTIVE | TERMINAL | WAITING == STATES
    assert not ACTIVE & TERMINAL
    assert not WAITING & TERMINAL
    assert "backoff" in ACTIVE and "observing" in ACTIVE


def dig_counts() -> dict[str, int]:
    counts = {}
    for path in sorted((ROOT / "spark_sweep").glob("*.py")):
        count = sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dig"
            for node in ast.walk(ast.parse(path.read_text()))
        )
        if count:
            counts[str(path.relative_to(ROOT))] = count
    return counts


def test_untyped_dig_access_only_falls() -> None:
    baseline = json.loads((ROOT / "tools/sweep-dig-baseline.json").read_text())
    counts = dig_counts()
    assert counts == baseline, (
        "dig access changed: remove new calls or tighten the baseline after a decrease"
    )


@pytest.mark.parametrize("word", sorted(STATES))
def test_fake_can_emit_every_lifecycle_word(tmp_path: Path, word: str) -> None:
    from sweep_fakes import FakeModel, FakeRecipe, make_sweep

    sweep, fleet, _ = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    fleet.observations.append((("recipe", "progress"), 0, {"id": "op", "state": word}))
    assert sweep.vk.call("recipe", "progress", "op")["state"] == word
