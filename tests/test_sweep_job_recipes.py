"""Job recipes are always tested; an older platform-skip entry returns to pending."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.run import IDENTICAL_EXIT_LIMIT, PLATFORM_SKIP


def _rows(tmp_path: Path, key: str) -> list[dict]:
    path = tmp_path / "results.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r.get("recipe") == key]


def test_job_recipe_is_pending_without_any_flag(tmp_path: Path) -> None:
    job = FakeRecipe("ocr", adapter="artifact-job")
    sweep, _, _ = make_sweep(tmp_path, [job], [FakeModel("m1")])
    sweep.refresh_catalog()
    assert sweep.state.status(job.key) == "pending"
    assert [r.key for r in sweep.pending()] == [job.key]


def test_an_older_platform_skip_returns_to_pending(tmp_path: Path) -> None:
    job = FakeRecipe("ocr", adapter="artifact-job")
    sweep, _, _ = make_sweep(tmp_path, [job], [FakeModel("m1")])
    sweep.refresh_catalog()
    sweep.state.recipes[job.key].update(
        status="skipped",
        reason="job recipes are not loadable through a fleet profile on this platform yet",
        reason_kind=PLATFORM_SKIP,
    )
    sweep.refresh_catalog()
    entry = sweep.state.recipes[job.key]
    assert entry["status"] == "pending"
    assert "reason" not in entry and "reason_kind" not in entry
    assert [r.key for r in sweep.pending()] == [job.key]


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


def test_a_job_recipe_passes_through_one_artifact_job(tmp_path: Path) -> None:
    job = FakeRecipe("ocr", adapter="artifact-job")
    sweep, fleet, _ = make_sweep(tmp_path, [job], [FakeModel("m1")])
    assert sweep.run() == 0
    assert sweep.state.status(job.key) == "passed"
    assert [step for step, _ in fleet.job_calls] == ["create", "submit", "download"]
    (row,) = [r for r in _rows(tmp_path, job.key) if r["status"] == "passed"]
    (case,) = row["result"]["cases"]
    assert row["result"]["kind"] == "job"
    assert case["outputs"][0]["name"] == "out.png"
    assert "not_evaluated" not in case


def test_three_identical_untyped_job_create_replies_defer_and_the_next_recipe_runs(
    tmp_path: Path,
) -> None:
    """vonkctl's generic ``control.api_error`` carries no typed wait: it is bounded, not endless."""
    bad = FakeRecipe("bad", adapter="artifact-job")
    good = FakeRecipe("good", adapter="artifact-job")
    sweep, fleet, _ = make_sweep(tmp_path, [bad, good], [FakeModel("m1")])
    detail = (
        "document does not match the canonical ArtifactJobCreate contract: "
        "$.timeout_seconds: violates maximum"
    )
    envelope = fleet._error("control.api_error", detail)[1]
    schema = Path(__file__).parent / "fixtures/vonkctl-error-v2.schema.json"
    jsonschema.Draft202012Validator(json.loads(schema.read_text())).validate(
        {**envelope, "schema_version": 2}
    )
    real = fleet._recipe_job

    def refuse_bad(a: list[str]) -> tuple[int, Any]:
        if a[0] == "create":
            run_id = a[a.index("--run") + 1]
            if any(r["run_id"] == run_id and r["alias"] == "bad" for r in fleet.runs):
                return fleet._error("control.api_error", detail)
        return real(a)

    fleet._recipe_job = refuse_bad  # type: ignore[method-assign]
    for _ in range(8):
        if sweep.run() == 0 and sweep.state.status(good.key) == "passed":
            break
    assert sweep.state.status(good.key) == "passed"
    deferred = [
        r
        for r in _rows(tmp_path, bad.key)
        if r.get("evidence", {}).get("identical_exits") == IDENTICAL_EXIT_LIMIT
    ]
    assert len(deferred) == 1
    assert deferred[0]["disposition"] == "fail"
    assert sweep.state.status(bad.key) == "deferred"
