"""The sweep proves a recipe runs; a wrong answer is a quality note, not a failure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy
from spark_sweep.issues import file_issues
from spark_sweep.report import (
    build_report,
    export_evidence,
    quality_notes,
    render_report,
)


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str = "a") -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _smoke_line(tmp_path: Path) -> dict:
    return next(
        json.loads(x)
        for x in (tmp_path / "results.jsonl").read_text().splitlines()
        if '"step": "smoke"' in x or '"step":"smoke"' in x
    )


def test_a_wrong_but_well_formed_answer_runs_with_a_quality_note(
    tmp_path: Path, gateway: Gateway
) -> None:
    gateway.messages["a"] = {"content": "305"}
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    entry = _entry(sweep)
    assert entry["status"] == "passed"
    assert entry["quality"] == [
        {"case": "A391", "expected": "391", "got": "305", "ok": False}
    ]
    line = _smoke_line(tmp_path)
    assert line["status"] == "passed"
    assert line["result"]["quality"] == [
        {"case": "A391", "expected": "391", "got": "305", "ok": False}
    ]


def test_the_expected_answer_leaves_no_note(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    assert _entry(sweep)["status"] == "passed"
    assert "quality" not in _entry(sweep)
    [check] = _smoke_line(tmp_path)["result"]["quality"]
    assert check["ok"] is True


@pytest.mark.parametrize(
    "message",
    [
        {"content": "The answer is 391"},
        {"content": ""},
        {"content": None},
        {"content": "391", "reasoning_content": "<think>working it out</think>"},
    ],
    ids=["prose", "empty", "no-content", "leaked-think-tag"],
)
def test_a_malformed_or_leaking_reply_still_fails_the_recipe(
    tmp_path: Path, gateway: Gateway, message: dict
) -> None:
    gateway.messages["a"] = message
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    entry = _entry(sweep)
    assert (entry["status"], entry["phase"], entry["failure_class"]) == (
        "failed",
        "smoke",
        "smoke-assertion",
    )


def test_quality_notes_are_reported_apart_from_failures(
    tmp_path: Path, gateway: Gateway
) -> None:
    gateway.messages["a"] = {"content": "305"}
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    recipes = sweep.state.recipes
    assert list(quality_notes(recipes)) == ["vonk-forge/a"]
    page = (tmp_path / "status.md").read_text()
    assert "passed 1 (1 with quality notes) - failed 0" in page
    assert "vonk-forge/a: 1 quality note" in page
    report = build_report(recipes)
    text = render_report(report)
    assert "ran; 1 quality note" in text
    assert "vonk-forge/a A391: expected 391, got 305" in text
    assert report["failure_clusters"] == []


def test_quality_notes_file_no_issue(tmp_path: Path, gateway: Gateway) -> None:
    gateway.messages["a"] = {"content": "305"}
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    calls: list[list[str]] = []

    def gh(argv):
        calls.append(list(argv))
        return 0, "[]"

    assert file_issues(sweep.state.recipes, "o/r", gh) == []
    assert not any(c[:2] == ["issue", "create"] for c in calls)


def test_evidence_export_keeps_the_quality_notes(
    tmp_path: Path, gateway: Gateway
) -> None:
    gateway.messages["a"] = {"content": "305"}
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    [path] = export_evidence(
        sweep.log.entries(), tmp_path / "evidence", ["vonk-forge/a"]
    )
    [line] = [json.loads(x) for x in path.read_text().splitlines()]
    assert line["result"]["quality"][0]["got"] == "305"


def test_a_recipe_that_failed_on_an_answer_under_the_old_rules_is_requeued(
    tmp_path: Path, gateway: Gateway
) -> None:
    gateway.messages["a"] = {"content": "305"}
    recipe = FakeRecipe("a")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    # As the old sweep left it: the exact answer failed the smoke case.
    entry = _entry(sweep)
    entry.update(
        status="failed",
        phase="smoke",
        failure_class="smoke-assertion",
        error="path.regex failed at choices.0.message.content: got '305'",
    )
    entry.pop("smoke_rules", None)
    entry.pop("quality", None)
    sweep.state.save()
    again, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    assert again.run() == 0
    assert _entry(again)["status"] == "passed"
    assert _entry(again)["quality"][0]["got"] == "305"
    requeue = [
        json.loads(x)
        for x in (tmp_path / "results.jsonl").read_text().splitlines()
        if '"requeue"' in x
    ]
    assert [x["reason"] for x in requeue] == [
        "smoke rules changed: answer quality no longer fails a recipe"
    ]


def test_a_functional_failure_under_the_new_rules_is_not_requeued_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    gateway.messages["a"] = {"content": "no number here"}
    recipe = FakeRecipe("a")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    assert _entry(sweep)["smoke_rules"] == policy.SMOKE_RULES
    again, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    again.run()
    assert not [
        x
        for x in (tmp_path / "results.jsonl").read_text().splitlines()
        if '"requeue"' in x
    ]
    assert _entry(again)["status"] == "failed"
