"""A load that copies and then waits learns a positive engine load time.

A copying tick that also reported a wait was counted in both ``copy_seconds`` and
``waited_seconds``; ``learn()`` subtracts both, so every learned vLLM load time was 0.0
and every no-progress window stayed at the floor.
"""

from pathlib import Path

from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import ControllerWait, SweepWait, SweepWaitReason

from spark_sweep.lifecycle import LifecycleState


def test_copy_ticks_are_not_also_counted_as_waited(tmp_path: Path) -> None:
    sweep, _, clock = make_sweep(tmp_path, [FakeRecipe("copy")], [FakeModel("m1")])
    key = "owned-copy"
    slot = {
        "request_key": key,
        "phase": "loading",
        "alias": "copy",
        "node_ids": [],
        "started_at": clock.now(),
        "last_tick": clock.now(),
        "deadline": clock.now() + 10_000,
    }
    wait = SweepWait(
        reason=SweepWaitReason.CONTROLLER_RETRY,
        controller=ControllerWait(reason="receipt-missing"),
    ).model_dump(mode="json")
    load = {
        "copying": True,
        "state": LifecycleState.RUNNING,
        "signal": {"signature": [1], "wait": wait},
    }
    sweep.state.data["loads"][key] = load
    clock.sleep(300)
    sweep._observe_load(slot, clock.now())  # 300 s of copying that also reports a wait
    load["copying"] = False
    load["signal"] = {"signature": [2], "wait": wait}
    clock.sleep(100)
    sweep._observe_load(slot, clock.now())  # 100 s of waiting only
    load["signal"] = {"signature": [3], "wait": None}
    clock.sleep(50)
    sweep._observe_load(slot, clock.now())  # 50 s of the engine itself starting
    assert slot["copy_seconds"] == 300
    assert slot["waited_seconds"] == 100
    # The engine's own load time, as the sweep learns it, is positive.
    assert sweep._engine_load_seconds(slot, clock.now()) == 50
