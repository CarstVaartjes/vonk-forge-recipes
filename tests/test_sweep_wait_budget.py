"""All wait reasons share a finite deadline and evidence-bearing lane release."""

from __future__ import annotations

import ast
from concurrent.futures import Future
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import SweepWait, SweepWaitCode, SweepWaitReason

from spark_sweep import verdict
from spark_sweep.report import build_status, write_status

ROOT = Path(__file__).resolve().parents[1]


def wait_bypasses(source: str, *, primitive: bool = False) -> list[int]:
    bad = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                else []
            )
            writes = any(
                isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and t.slice.value == "waiting_on"
                for t in targets
            )
            if isinstance(node, ast.Call):
                writes |= any(k.arg == "waiting_on" for k in node.keywords)
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "update",
                    "setdefault",
                }:
                    writes |= any(
                        isinstance(n, ast.Constant) and n.value == "waiting_on"
                        for arg in node.args
                        for n in ast.walk(arg)
                    )
            if isinstance(node, ast.Dict):
                writes |= any(
                    isinstance(k, ast.Constant)
                    and k.value == "waiting_on"
                    and isinstance(v, (ast.Constant, ast.JoinedStr))
                    for k, v in zip(node.keys, node.values)
                )
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "waiting_on"
                and isinstance(node.ctx, ast.Store)
            ):
                writes = True
            if writes and not (primitive and function.name == "bounded_wait"):
                bad.append(getattr(node, "lineno", function.lineno))
    return bad


def test_every_waiting_on_producer_uses_the_primitive() -> None:
    assert wait_bypasses("def bad(w):\n    w['waiting_on'] = 'new reason'\n")
    assert wait_bypasses("def bad(w):\n    w.update(waiting_on='new reason')\n")
    assert wait_bypasses("def bad(w):\n    w.update({'waiting_on': 'new reason'})\n")
    assert wait_bypasses("def bad():\n    return {'waiting_on': 'new reason'}\n")
    for path in (ROOT / "spark_sweep").rglob("*.py"):
        assert not wait_bypasses(
            path.read_text(), primitive=path.name == "verdict.py"
        ), str(path)
        # Every reason must be a contract enum member, never an invented string.
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "SweepWait"
            ):
                reason = next(k.value for k in node.keywords if k.arg == "reason")
                assert (
                    isinstance(reason, ast.Attribute)
                    and isinstance(reason.value, ast.Name)
                    and reason.value.id == "SweepWaitReason"
                )


@pytest.mark.parametrize("reason", list(SweepWaitReason))
def test_each_wait_releases_lane_and_requeues_after_budget(
    tmp_path: Path, reason: SweepWaitReason
) -> None:
    recipe = FakeRecipe("a", local="cached")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    slot = sweep.state.slots[recipe.key]
    start = clock.now()
    slot.update(phase="smoking", ready_at=start, smoke_started_at=start)
    wait = SweepWait(reason=reason, detail="injected dependency hold")
    assert not sweep._wait_slot(recipe.key, slot, wait)
    budget = slot["deadline"] - slot["started_at"]
    # Another reason and apparent progress cannot renew the original deadline.
    clock.sleep(budget / 2)
    assert not sweep._wait_slot(recipe.key, slot, wait)
    clock.sleep(budget / 2 + 1)
    write_status(sweep)
    assert "OVER BUDGET" in (tmp_path / "status.md").read_text()
    assert sweep._wait_slot(recipe.key, slot, wait)
    sweep.enqueue_finished_cleanup()
    assert recipe.key not in sweep.state.slots
    assert sweep.state.status(recipe.key) == "pending"
    entry = sweep.state.entry(recipe.key)
    assert entry.get("attempts", 0) == 0
    assert entry["not_before"] > clock.now()
    row = list(sweep.log.entries())[-1]
    assert row["code"] == SweepWaitCode.BUDGET_EXHAUSTED
    assert row["evidence"]["wait"]["reason"] == reason
    assert row["evidence"]["observed_at"] > row["evidence"]["deadline"]
    assert row["controller_document"]
    assert not any(build_status(sweep)["lanes"].values())


@pytest.mark.parametrize("reason", list(SweepWaitReason))
def test_wait_reason_changes_do_not_extend_budget(
    reason: SweepWaitReason,
) -> None:
    watch = {}
    assert (
        verdict.progress_verdict(
            verdict.Signal((1,), SweepWait(reason=reason)), watch, window=10, now=0
        )
        is verdict.Progress.WAITING
    )
    assert (
        verdict.progress_verdict(
            verdict.Signal((1,), SweepWait(reason=SweepWaitReason.OBSERVATION)),
            watch,
            window=10,
            now=11,
        )
        is verdict.Progress.WAIT_EXPIRED
    )


def test_sse_heartbeats_cannot_hold_smoking_lane_forever(tmp_path: Path) -> None:
    recipe = FakeRecipe("a", local="cached")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    slot = sweep.state.slots[recipe.key]
    slot.update(phase="smoking", ready_at=clock.now())
    # A running worker cannot be stopped by Future.cancel(); the lane must release
    # independently and signal the stream to close cooperatively.
    import threading

    future = Future()
    future.set_running_or_notify_cancel()
    stop = threading.Event()
    sweep.smoke_stops[recipe.key] = stop
    sweep.futures[recipe.key] = future
    sweep._advance_smoking(recipe.key, slot)
    clock.sleep(slot["deadline"] - slot["started_at"] + 1)
    sweep._advance_smoking(recipe.key, slot)
    sweep.enqueue_finished_cleanup()
    assert recipe.key not in sweep.state.slots
    assert stop.is_set() and not future.cancelled()
    assert (
        sweep.state.entry(recipe.key)["wait_deferral"]["wait"]["reason"]
        == SweepWaitReason.SMOKE_RESPONSE
    )


@pytest.mark.parametrize("reason", list(SweepWaitReason))
def test_loading_consumer_releases_each_expired_signal(
    tmp_path: Path, reason: SweepWaitReason
) -> None:
    recipe = FakeRecipe("held", local="cached", blocked=True)
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    slot = sweep.state.slots[recipe.key]
    load = sweep.state.data["loads"][slot["request_key"]]
    load["signal"] = {
        "signature": [],
        "wait": SweepWait(reason=reason).model_dump(mode="json"),
    }
    clock.sleep(slot["deadline"] - slot["started_at"] + 1)
    sweep._advance_loading(recipe.key, slot, clock.now())
    sweep.enqueue_finished_cleanup()
    assert recipe.key not in sweep.state.slots
    assert sweep.state.entry(recipe.key)["wait_deferral"]["wait"]["reason"] == reason


def test_unavailable_observation_releases_even_during_a_blind_pass(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("held", local="cached", blocked=True)
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    slot = sweep.state.slots[recipe.key]
    sweep.blind = True
    clock.sleep(slot["deadline"] - slot["started_at"] + 1)
    sweep._advance_loading(recipe.key, slot, clock.now())
    assert slot["phase"] == "finished"
    assert (
        sweep.state.entry(recipe.key)["wait_deferral"]["wait"]["reason"]
        == SweepWaitReason.OBSERVATION
    )


def test_expired_stream_stops_on_the_next_heartbeat(monkeypatch) -> None:
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace

    from spark_sweep import smoke

    stopped = threading.Event()
    closed = []

    def heartbeats():
        yield b": heartbeat\n"
        stopped.set()
        yield b": heartbeat\n"
        raise AssertionError("the expired stream was still consumed")

    @contextmanager
    def response(*_args):
        try:
            lines = heartbeats()
            yield SimpleNamespace(read1=lambda _size: next(lines, b""))
        finally:
            closed.append(True)

    monkeypatch.setattr(smoke, "open_with_body", response)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        smoke.stream_probe("http://fake", "alias", smoke.HttpConfig(cancelled=stopped))
    assert closed == [True]


def test_replanned_cases_return_newest_first_after_the_delay(tmp_path: Path) -> None:
    from vonk_forge_contracts.sweep import SweepWaitExpiry

    recipes = [
        FakeRecipe("old", local="cached", updated="2026-10-01T00:00:00Z"),
        FakeRecipe("new", local="cached", updated="2026-10-20T00:00:00Z"),
    ]
    sweep, _, clock = make_sweep(tmp_path, recipes, [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = [r.key for r in recipes]
    sweep.cached_models = sweep._cached_models()
    for recipe in recipes:
        sweep.state.entry(recipe.key).update(
            not_before=clock.now() + sweep.cfg.defer_delay,
            wait_deferral=SweepWaitExpiry(
                wait=SweepWait(reason=SweepWaitReason.ADMISSION),
                started_at=0,
                deadline=1,
                observed_at=2,
            ).model_dump(mode="json"),
        )
    assert not sweep.ready_candidates()
    clock.sleep(sweep.cfg.defer_delay)
    assert [r.key for r in sweep.ready_candidates()] == [
        r.key for r in reversed(recipes)
    ]
