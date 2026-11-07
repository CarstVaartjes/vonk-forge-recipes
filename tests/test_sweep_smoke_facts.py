"""The smoke's recipe facts are read on the smoke's worker thread, once per content.

Resolving the facts reads the recipe document from the Controller (``vonkctl recipe
detail``, up to a minute). That wait belongs to the lane's worker thread: the tick that
schedules every other lane never blocks on it, and the pass that follows reuses the read.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import memory, run
from spark_sweep.vonkctl import Reply


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def test_the_facts_are_resolved_off_the_tick_thread(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = FakeRecipe("r")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    main = threading.current_thread()
    seen: list[threading.Thread] = []
    release = threading.Event()

    def slow(*_: Any) -> None:
        seen.append(threading.current_thread())
        release.wait(5)  # a slow `recipe detail`

    monkeypatch.setattr(run.Sweep, "_chat_facts", slow)
    blocked_ticks = 0
    for _ in range(4000):
        sweep.tick()
        clock.sleep(sweep.cfg.poll_seconds)
        if seen:
            blocked_ticks += 1
            if blocked_ticks >= 20:
                break
    release.set()
    # While the read was outstanding the scheduler kept ticking, on its own thread.
    assert blocked_ticks >= 20
    assert seen and seen[0] is not main
    for _ in range(4000):
        sweep.tick()
        if sweep.state.status(recipe.key) == "passed":
            break
        clock.sleep(sweep.cfg.poll_seconds)
    assert sweep.state.status(recipe.key) == "passed"


def test_one_read_of_the_recipe_document_serves_the_smoke_and_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe_row = FakeRecipe("r")
    sweep, _, _ = make_sweep(tmp_path, [recipe_row], [FakeModel("m1")])
    reads: list[tuple[str, ...]] = []

    def counting(*argv: str, **_: Any) -> Reply:
        reads.append(argv)
        return Reply(argv, 0, {"document": {}}, "")

    sentinel = object()
    monkeypatch.setattr(sweep.vk, "run", counting)
    monkeypatch.setattr(memory, "ran_recipe", lambda *_: sentinel)

    class Row:
        key = "vonk-forge/r"
        content_sha256 = "c1"

    assert sweep._ran_recipe(Row()) is sentinel  # type: ignore[arg-type]
    assert sweep._ran_recipe(Row()) is sentinel  # type: ignore[arg-type]
    assert len(reads) == 1
