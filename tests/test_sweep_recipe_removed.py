"""A recipe that leaves the complete catalog is skipped with its reason, never retried as infra."""

from __future__ import annotations

import json
from pathlib import Path

from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep import policy
from spark_sweep.prefetch import PrefetchConfig
from spark_sweep.verdict import Disposition, disposition


def _rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "results.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_removed_recipe_is_a_skip_disposition() -> None:
    failure = policy.classify("catalog", "catalog.recipe_removed", "absent")
    assert disposition(failure) is Disposition.SKIP


def test_removed_recipe_drops_its_download_without_infra_backoff(
    tmp_path: Path,
) -> None:
    gone, kept = FakeRecipe("gone"), FakeRecipe("kept")
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [gone, kept],
        [FakeModel("m1")],
        prefetch=PrefetchConfig(pin_profile=None),
    )
    sweep.refresh_catalog()
    sweep.prefetcher._request(sweep.recipes[gone.key], "model")
    assert sweep.state.downloads[gone.key]["state"] not in ("succeeded", "failed")
    pause = sweep.prefetcher.pause_until

    fleet.ops.clear()  # the fake's library cannot list an op of a gone recipe
    del fleet.recipes[gone.key]
    sweep.refresh_catalog()

    assert "download" not in sweep.state.data["infra"]
    assert sweep.prefetcher.pause_until == pause
    assert gone.key not in sweep.state.downloads
    assert gone.key not in sweep.state.recipes
    skipped = [r for r in _rows(tmp_path) if r.get("recipe") == gone.key]
    assert [(r["status"], r["code"]) for r in skipped] == [
        ("skipped", "catalog.recipe_removed")
    ]

    fleet.recipes[gone.key] = gone  # back in a later catalog: picked up like a new one
    sweep.refresh_catalog()
    assert sweep.state.status(gone.key) == "pending"


def test_removed_recipe_in_a_lane_is_skipped_not_deferred(tmp_path: Path) -> None:
    gone = FakeRecipe("gone")
    sweep, fleet, clock = make_sweep(tmp_path, [gone], [FakeModel("m1")])
    sweep.refresh_catalog()
    sweep.state.slots[gone.key] = {
        "request_key": "lane-request",
        "phase": "loading",
        "alias": "gone-alias",
        "node_ids": ["spk_a"],
        "started_at": clock.now() + 1,
    }

    del fleet.recipes[gone.key]
    sweep.refresh_catalog()

    entry = sweep.state.recipes[gone.key]
    assert entry["status"] == "skipped"
    assert "absent from the complete catalog" in entry["reason"]
    assert entry.get("attempts", 0) == 0
    assert sweep.state.slots[gone.key]["phase"] == "finished"
    rows = [r for r in _rows(tmp_path) if r.get("recipe") == gone.key]
    assert [(r["status"], r["disposition"]) for r in rows] == [("skipped", "skip")]

    fleet.recipes[gone.key] = gone
    sweep.refresh_catalog()
    assert sweep.state.status(gone.key) == "pending"
    assert "reason" not in sweep.state.recipes[gone.key]
