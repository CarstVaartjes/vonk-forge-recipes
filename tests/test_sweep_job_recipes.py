"""Job recipes are skipped before any lane is spent, and the skip heals once they load."""

from __future__ import annotations

import json
from pathlib import Path

from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.run import JOB_SKIP_CODE, PLATFORM_SKIP


def _rows(tmp_path: Path, key: str) -> list[dict]:
    path = tmp_path / "results.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r.get("recipe") == key]


def test_job_recipe_is_skipped_without_a_lane_and_heals(tmp_path: Path) -> None:
    job, chat = FakeRecipe("ocr", adapter="artifact-job"), FakeRecipe("chat")
    sweep, fleet, _ = make_sweep(tmp_path, [job, chat], [FakeModel("m1")])
    sweep.refresh_catalog()

    entry = sweep.state.recipes[job.key]
    assert entry["status"] == "skipped"
    assert entry["reason_kind"] == PLATFORM_SKIP
    assert entry.get("attempts", 0) == 0
    assert sweep.state.status(chat.key) == "pending"
    assert [r.key for r in sweep.pending()] == [chat.key]
    assert job.key not in sweep.state.slots

    sweep.refresh_catalog()  # idempotent: one row, however many passes
    rows = _rows(tmp_path, job.key)
    assert [(r["status"], r.get("code")) for r in rows] == [("skipped", JOB_SKIP_CODE)]

    fleet.recipes[job.key] = FakeRecipe("ocr")  # the platform loads job recipes now
    sweep.refresh_catalog()
    entry = sweep.state.recipes[job.key]
    assert entry["status"] == "pending"
    assert "reason" not in entry and "reason_kind" not in entry


def test_job_recipe_in_a_live_lane_is_left_to_finish(tmp_path: Path) -> None:
    job = FakeRecipe("ocr", adapter="artifact-job")
    sweep, _, clock = make_sweep(tmp_path, [job], [FakeModel("m1")])
    sweep.state.slots[job.key] = {
        "request_key": "lane-request",
        "phase": "loading",
        "alias": "ocr",
        "node_ids": ["spk_a"],
        "started_at": clock.now() + 1,
    }
    sweep.refresh_catalog()
    assert sweep.state.status(job.key) == "pending"
    assert job.key not in [r.key for r in sweep.pending()]
