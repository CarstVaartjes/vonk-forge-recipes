"""The sweep never idles while ready work fits a free lane.

On hardware a dual's admission stalled; its lane failed and cancelled the application detached,
but the Controller kept holding it, so the sweep's load record stayed `running` forever and no
lane was scheduled again for over an hour.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _orphan(sweep) -> None:
    sweep.state.data["loads"]["stale-request"] = {
        "request_key": "stale-request",
        "app_id": "app-stale",
        "seq": 0,
        "submitted_at": 0.0,
        "state": "running",
    }


def test_a_load_whose_lanes_ended_does_not_block_scheduling(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a", ("m1",)), FakeRecipe("b", ("m2",))]
    sweep, fleet, _ = make_sweep(
        tmp_path, recipes, [FakeModel("m1"), FakeModel("m2")], gateway=gateway
    )
    _orphan(sweep)
    assert sweep.run() == 0
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert sweep.state.data["loads"] == {}
    cancels = [c for _, c in fleet.calls if c[:2] == ("profile", "cancel")]
    assert any("app-stale" in c for c in cancels)  # ours, unsettled: released


def test_a_load_with_a_live_lane_is_kept(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")], gateway=gateway
    )
    _orphan(sweep)
    sweep.state.slots["a"] = {
        "request_key": "stale-request",
        "phase": "loading",
        "node_ids": [],
    }
    sweep.release_orphan_load()
    assert sweep.state.data["loads"]


def test_idle_with_ready_work_raises_an_alert(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")], gateway=gateway
    )
    sweep.sparks = lambda: [SimpleNamespace(id="spk_a")]  # type: ignore[method-assign]
    sweep.ready_candidates = lambda: [object()]  # type: ignore[method-assign]
    now = sweep.clock.now()
    sweep.check_idle(now)
    assert not sweep.state.data.get("alerts")
    sweep.check_idle(now + sweep.cfg.idle_alert_seconds + 1)
    assert sweep.state.data["alerts"]
    assert "IDLE WITH READY WORK" in sweep.state.data["alerts"][0]
    sweep.ready_candidates = list  # type: ignore[method-assign]
    sweep.check_idle(now + 2 * sweep.cfg.idle_alert_seconds)
    assert sweep.state.data["alerts"] == []
