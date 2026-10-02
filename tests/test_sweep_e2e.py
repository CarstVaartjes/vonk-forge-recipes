"""End to end: the whole sweep against a fake vonkctl, a fake clock and a loopback gateway."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_two_lanes_share_a_model_and_all_pass(tmp_path: Path, gateway: Gateway) -> None:
    recipes = [
        FakeRecipe("a-vllm", ("m1",)),
        FakeRecipe("a-sglang", ("m1",), engine="sglang"),
        FakeRecipe("b-vllm", ("m2",)),
    ]
    models = [FakeModel("m1"), FakeModel("m2")]
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    assert sweep.run() == 0
    assert {k: e["status"] for k, e in sweep.state.recipes.items()} == {
        "vonk-forge/a-vllm": "passed",
        "vonk-forge/a-sglang": "passed",
        "vonk-forge/b-vllm": "passed",
    }
    lines = _results(tmp_path / "results.jsonl")
    assert all(line["step"] == "smoke" and line["status"] == "passed" for line in lines)
    first = lines[0]
    assert first["authority_id"] == "test-authority"
    assert first["result"]["perf"]["tokens_per_second"] > 0
    assert first["result"]["perf"]["ttft_ms"] >= 0
    assert [c["case_id"] for c in first["result"]["cases"]] == ["M0", "A391"]
    assert first["content_sha256"] == "c1" and first["library_commit"] == "lib-commit-1"
    # The model was downloaded once, then its sibling pulled only an image.
    downloads = [c for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert len(downloads) == 3
    # Owner profiles are never written; every write used the sweep or pin profile.
    assert not any(
        p in (1, 2, 3)
        for p, c in fleet.calls
        if c[0] == "profile" and c[1] in ("add", "remove", "load", "cancel")
    )
    assert fleet.writes(10) and fleet.writes(13)


def test_lanes_run_in_parallel_one_per_spark(tmp_path: Path, gateway: Gateway) -> None:
    recipes = [
        FakeRecipe("x", ("m1",), memory=100 * 10**9),
        FakeRecipe("y", ("m2",), memory=100 * 10**9),
    ]
    models = [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")]
    for r in recipes:
        r.local = "cached"
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models, gateway=gateway)
    seen = []
    clock.hooks.append(
        lambda _t: seen.append(
            sorted(
                (s["spark_names"][0], s["phase"]) for s in sweep.state.slots.values()
            )
        )
    )
    sweep.run()
    # Both Sparks held a recipe at the same time (two lanes, 100 GB each cannot share one Spark).
    assert any(
        len(snapshot) == 2 and snapshot[0][0] != snapshot[1][0] for snapshot in seen
    )
    # One load carried both lanes; the only other load is the final stop.
    writes = [
        c[1]
        for p, c in fleet.calls
        if p == 10 and c[0] == "profile" and c[1] in ("add", "remove", "load")
    ]
    first_load = writes.index("load")
    assert writes[:first_load].count("add") == 2
    assert writes.count("load") == 3  # review, load, stop at the end
