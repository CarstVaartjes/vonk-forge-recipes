"""Fault injection proves evidence transport cannot hold hardware lanes."""

import json
import subprocess
from pathlib import Path

import pytest
from test_qualification_status import historical

from spark_sweep import evidence_publisher as publisher


def seed_catalog(root):
    source = Path(__file__).resolve().parents[1] / "recipes/qwen3-5-9b-vllm-single.json"
    document = json.loads(source.read_text())
    document["identity"]["slug"] = "example"
    (root / "recipes").mkdir(exist_ok=True)
    (root / "recipes/example.json").write_text(json.dumps(document))


def test_sync_batches_one_pr_requests_auto_merge_and_replans_after_push_race(
    tmp_path, monkeypatch
):
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps(historical()) + "\n")
    calls = []
    fail_push = True
    pushed = False

    def command(cwd, *args, timeout=60):
        nonlocal fail_push, pushed
        calls.append(args)
        assert timeout <= 120
        if args[:2] == ("git", "clone"):
            (Path(args[-1]) / ".git").mkdir()
            seed_catalog(Path(args[-1]))
        if args[:3] == ("git", "diff", "origin/main"):
            return "evidence diff"
        if args[:3] == ("git", "diff", "--cached"):
            return "qualification/evidence/vonk-forge/example.json"
        if args[:2] == ("git", "push") and fail_push:
            fail_push = False
            raise subprocess.CalledProcessError(1, args, stderr="push raced")
        if args[:2] == ("git", "push"):
            pushed = True
        if args[:3] == ("gh", "pr", "list"):
            return '[{"number": 123}]' if pushed else "[]"
        return ""

    monkeypatch.setattr(publisher, "command", command)
    with pytest.raises(subprocess.CalledProcessError):
        publisher.sync(history, tmp_path / "work")
    publisher.sync(history, tmp_path / "work")
    assert sum(c[:2] == ("git", "clone") for c in calls) == 2
    assert not any("--force" in c for c in calls)
    assert not any(c[:3] == ("gh", "pr", "create") for c in calls)
    assert any(c[:4] == ("gh", "pr", "merge", "123") and "--auto" in c for c in calls)
    assert history.read_text() == json.dumps(historical()) + "\n"


def test_empty_batch_never_opens_pr(tmp_path, monkeypatch):
    history = tmp_path / "results.jsonl"
    history.write_text("")
    calls = []

    def command(cwd, *args, **kwargs):
        calls.append(args)
        if args[:2] == ("git", "clone"):
            seed_catalog(Path(args[-1]))
        return "[]" if args[:3] == ("gh", "pr", "list") else ""

    monkeypatch.setattr(publisher, "command", command)
    publisher.sync(history, tmp_path / "work")
    assert not any(
        c[:3] in (("gh", "pr", "create"), ("gh", "pr", "merge"))
        or c[:2] == ("git", "push")
        for c in calls
    )


def test_lane_tick_never_waits_on_github_and_timeout_is_visible(tmp_path, monkeypatch):
    processes = []

    class Process:
        code = None
        killed = False

        def __init__(self, *args, **kwargs):
            processes.append(self)

        def poll(self):
            return self.code

        def kill(self):
            self.killed = True
            self.code = -9

        def wait(self, timeout):
            assert timeout <= 1
            return self.code

    monkeypatch.setattr(publisher.subprocess, "Popen", Process)
    worker = publisher.EvidencePublisher(
        tmp_path / "results.jsonl", tmp_path / "publisher", Path("worker")
    )
    worker.tick(0)
    assert len(processes) == 1
    worker.tick(100)
    assert len(processes) == 1 and not processes[0].killed
    worker.tick(241)
    assert processes[0].killed and "240s" in worker.message
    worker.tick(242)
    assert "re-plan" in worker.message
    worker.tick(541)
    assert len(processes) == 1
    worker.tick(542)
    assert len(processes) == 2
    processes[-1].code = 0
    worker.tick(543)


def test_evidence_merge_conflict_unions_both_histories(tmp_path, monkeypatch):
    from datetime import timedelta

    from test_qualification_status import KEY, NOW, evidence, record

    from spark_sweep.qualification import read_evidence, write_evidence

    history = tmp_path / "results.jsonl"
    row = historical()
    row["recorded_at"] = (NOW + timedelta(days=2)).isoformat()
    history.write_text(json.dumps(row) + "\n")
    observed = []
    lists = 0

    def command(cwd, *args, **kwargs):
        nonlocal lists
        if args[:2] == ("git", "clone"):
            clone = Path(args[-1])
            (clone / ".git").mkdir()
            seed_catalog(clone)
            write_evidence(clone, evidence(record()))
        if args[:2] == ("git", "ls-remote"):
            return "remote-branch"
        if "--diff-filter=U" in args:
            return "qualification/evidence/vonk-forge/example.json"
        if args[:3] == ("git", "diff", "origin/main"):
            observed.extend(read_evidence(cwd, KEY).records)
            return "changed evidence"
        if args[:3] == ("git", "diff", "--cached"):
            return "qualification/evidence/vonk-forge/example.json"
        if args[:3] == ("gh", "pr", "list"):
            lists += 1
            return "[]" if lists == 1 else '[{"number":123}]'
        return ""

    def merge(args, *, cwd, **kwargs):
        assert args[:2] == ["git", "merge"]
        write_evidence(cwd, evidence(record(day=1)))
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(publisher, "command", command)
    monkeypatch.setattr(publisher.subprocess, "run", merge)
    publisher.sync(history, tmp_path / "work")
    assert len(observed) == 3
    assert [r.tested_at for r in observed] == [
        NOW + timedelta(days=d) for d in range(3)
    ]


def test_publisher_spawn_failure_is_visible_and_never_escapes_to_lanes(
    tmp_path, monkeypatch
):
    attempts = []

    def refused(*args, **kwargs):
        attempts.append(args)
        raise OSError("publisher unavailable")

    monkeypatch.setattr(publisher.subprocess, "Popen", refused)
    worker = publisher.EvidencePublisher(
        tmp_path / "results.jsonl", tmp_path / "publisher", Path("worker")
    )
    worker.tick(0)
    assert "unavailable" in worker.message and "300s" in worker.message
    assert worker.process is None
    worker.tick(299)
    assert len(attempts) == 1
    worker.tick(300)
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "state",
    [
        publisher.SweepEvidenceMergeState.BLOCKED,
        publisher.SweepEvidenceMergeState.CLEAN,
        publisher.SweepEvidenceMergeState.UNKNOWN,
    ],
)
def test_open_pr_has_a_quiet_head_for_ci_and_auto_merge(tmp_path, monkeypatch, state):
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps(historical()) + "\n")
    before = history.read_bytes()
    calls = []

    def command(cwd, *args, **kwargs):
        calls.append(args)
        if args[:3] == ("gh", "pr", "list"):
            return '[{"number":123}]'
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps({"number": 123, "mergeStateStatus": state})
        return ""

    monkeypatch.setattr(publisher, "command", command)
    assert publisher.sync(history, tmp_path / "work") is False
    assert not any(c[0] == "git" for c in calls)
    assert any(c[:3] == ("gh", "pr", "merge") and "--auto" in c for c in calls)
    assert history.read_bytes() == before


def test_sync_removes_withdrawn_evidence_despite_branch_conflict_and_old_history(
    tmp_path, monkeypatch
):
    from test_qualification_status import Evidence, evidence, record

    from spark_sweep.qualification import evidence_path, write_evidence

    key = "vonk-forge/withdrawn"
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps({**historical(), "recipe": key}) + "\n")
    observed = []
    calls = []
    lists = 0

    def command(cwd, *args, **kwargs):
        nonlocal lists
        calls.append(args)
        if args[:2] == ("git", "clone"):
            clone = Path(args[-1])
            (clone / ".git").mkdir()
            seed_catalog(clone)
            write_evidence(clone, evidence(record()))
            write_evidence(clone, Evidence(recipe_key=key, records=(record(),)))
        if args[:2] == ("git", "ls-remote"):
            return "remote-branch"
        if "--diff-filter=U" in args:
            return "qualification/evidence/vonk-forge/withdrawn.json"
        if args[:3] == ("git", "diff", "origin/main"):
            observed.append(evidence_path(cwd, key).exists())
            return "removed withdrawn evidence"
        if args[:3] == ("git", "diff", "--cached"):
            return "qualification/evidence/vonk-forge/withdrawn.json"
        if args[:3] == ("gh", "pr", "list"):
            lists += 1
            return "[]" if lists == 1 else '[{"number":123}]'
        return ""

    monkeypatch.setattr(publisher, "command", command)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1),
    )
    publisher.sync(history, tmp_path / "work")
    assert observed == [False]
    assert (
        "git",
        "rm",
        "--",
        "qualification/evidence/vonk-forge/withdrawn.json",
    ) in calls
