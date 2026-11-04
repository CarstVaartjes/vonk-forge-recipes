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
            raise subprocess.CalledProcessError(
                1, args, stderr="! [rejected] HEAD -> x (fetch first)"
            )
        if args[:2] == ("git", "push"):
            pushed = True
        if args[:3] == ("gh", "pr", "list"):
            return '[{"number": 123}]' if pushed else "[]"
        return ""

    monkeypatch.setattr(publisher, "command", command)
    first = publisher.sync(history, tmp_path / "work")
    assert first == publisher.SweepEvidenceExit.WAITING  # a race is transient
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
    assert (
        publisher.sync(history, tmp_path / "work")
        == publisher.SweepEvidenceExit.WAITING
    )
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


RED = [{"status": "COMPLETED", "conclusion": "FAILURE"}]
CANCELLED = [{"status": "COMPLETED", "conclusion": "CANCELLED"}]
COMMIT_STATUS = [{"state": "FAILURE"}]
PASSED = [
    {"status": "COMPLETED", "conclusion": "SUCCESS"},
    {"status": "COMPLETED", "conclusion": "SKIPPED"},
    {"status": "COMPLETED", "conclusion": "NEUTRAL"},
    {"state": "SUCCESS"},
]
PENDING = [{"status": "IN_PROGRESS", "conclusion": ""}]


def _open_pr(tmp_path, monkeypatch, rollup, behind_by):
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps(historical()) + "\n")
    calls = []

    def command(cwd, *args, **kwargs):
        calls.append(args)
        if args[:3] == ("gh", "pr", "list"):
            return '[{"number":123}]'
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps(
                {
                    "number": 123,
                    "mergeStateStatus": "BLOCKED",
                    "statusCheckRollup": rollup,
                }
            )
        if args[:2] == ("gh", "api"):
            return str(behind_by)
        if args[:2] == ("git", "clone"):
            raise RuntimeError("fresh plan reached")
        return ""

    monkeypatch.setattr(publisher, "command", command)
    return history, calls


def test_red_pr_with_moved_main_takes_the_fresh_plan(tmp_path, monkeypatch):
    history, calls = _open_pr(tmp_path, monkeypatch, RED, 3)
    with pytest.raises(RuntimeError, match="fresh plan reached"):
        publisher.sync(history, tmp_path / "work")
    assert not any(c[:3] == ("gh", "pr", "merge") for c in calls)


def test_red_pr_on_current_main_waits_and_says_so(tmp_path, monkeypatch, capsys):
    history, calls = _open_pr(tmp_path, monkeypatch, RED, 0)
    assert (
        publisher.sync(history, tmp_path / "work")
        == publisher.SweepEvidenceExit.FAILING_CHECK
    )
    assert not any(c[0] == "git" for c in calls)
    assert (
        "evidence PR #123 has a failing check; waiting for main to move"
        in capsys.readouterr().out
    )


def test_pending_pr_waits_as_before(tmp_path, monkeypatch, capsys):
    history, calls = _open_pr(tmp_path, monkeypatch, PENDING, 3)
    assert (
        publisher.sync(history, tmp_path / "work")
        == publisher.SweepEvidenceExit.WAITING
    )
    assert not any(c[0] == "git" or c[:2] == ("gh", "api") for c in calls)
    assert "awaiting CI/auto-merge" in capsys.readouterr().out


def test_lane_status_shows_a_failing_evidence_check(tmp_path, monkeypatch):
    class Process:
        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return publisher.SweepEvidenceExit.FAILING_CHECK

        def wait(self, timeout):
            return self.poll()

    monkeypatch.setattr(publisher.subprocess, "Popen", Process)
    worker = publisher.EvidencePublisher(
        tmp_path / "results.jsonl", tmp_path / "publisher", Path("worker")
    )
    worker.tick(0)
    worker.tick(1)
    assert worker.message == (
        "evidence PR has a failing check; waiting for main to move (see publisher.log)"
    )
    assert worker.next_attempt == 301


@pytest.mark.parametrize("rollup", [RED, CANCELLED, COMMIT_STATUS])
def test_cancelled_and_commit_status_failures_are_red(rollup, tmp_path, monkeypatch):
    history, _calls = _open_pr(tmp_path, monkeypatch, rollup, 0)
    assert (
        publisher.sync(history, tmp_path / "work")
        == publisher.SweepEvidenceExit.FAILING_CHECK
    )


def test_passed_and_pending_checks_are_not_red(tmp_path, monkeypatch):
    history, calls = _open_pr(tmp_path, monkeypatch, PASSED + PENDING, 3)
    assert (
        publisher.sync(history, tmp_path / "work")
        == publisher.SweepEvidenceExit.WAITING
    )
    assert not any(c[:2] == ("gh", "api") for c in calls)


def test_behind_pr_replans_without_asking_how_far_main_moved(tmp_path, monkeypatch):
    history, calls = _open_pr(tmp_path, monkeypatch, RED, 3)
    original = publisher.command

    def behind(cwd, *args, **kwargs):
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps(
                {"number": 123, "mergeStateStatus": "BEHIND", "statusCheckRollup": RED}
            )
        return original(cwd, *args, **kwargs)

    monkeypatch.setattr(publisher, "command", behind)
    with pytest.raises(RuntimeError, match="fresh plan reached"):
        publisher.sync(history, tmp_path / "work")
    assert not any(c[:2] == ("gh", "api") for c in calls)


def _publish_with_state(tmp_path, monkeypatch, repo_records, state_evidence):
    """Run one sync with mocked git/gh; return (state text, records the repo ends with)."""
    from test_qualification_status import KEY

    from spark_sweep.qualification import read_evidence, write_evidence

    history = tmp_path / "results.jsonl"
    history.write_text("")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {"qualification_evidence": {KEY: state_evidence.model_dump(mode="json")}}
        )
    )
    before = state.read_text()
    final = []
    listed = 0

    def command(cwd, *args, **kwargs):
        nonlocal listed
        if args[:2] == ("git", "clone"):
            clone = Path(args[-1])
            (clone / ".git").mkdir()
            seed_catalog(clone)
            if repo_records:
                write_evidence(clone, repo_records)
        if args[:3] == ("git", "diff", "origin/main"):
            final.extend(read_evidence(cwd, KEY).records)
            return "changed evidence"
        if args[:3] == ("git", "diff", "--cached"):
            return "qualification/evidence/vonk-forge/example.json"
        if args[:3] == ("gh", "pr", "list"):
            listed += 1
            return "[]" if listed == 1 else '[{"number":7}]'
        return ""

    monkeypatch.setattr(publisher, "command", command)
    publisher.sync(history, tmp_path / "work", state)
    assert state.read_text() == before  # the child only reads the sweep state
    return final


def test_resolved_state_only_record_is_published(tmp_path, monkeypatch):
    from test_qualification_status import B, evidence, record

    published = _publish_with_state(tmp_path, monkeypatch, None, evidence(record(B)))
    assert [r.identity for r in published] == [B]
    assert published[0].identity.models_resolved and published[0].identity.model_digests


def test_state_evidence_is_only_appended_never_rewritten(tmp_path, monkeypatch):
    from test_qualification_status import A, B, evidence, record

    old = record(A, day=0)
    published = _publish_with_state(
        tmp_path, monkeypatch, evidence(old), evidence(record(B, day=1))
    )
    assert published[0] == old  # existing repo record kept as it was
    assert [r.identity for r in published] == [A, B]
    # Re-publishing the same state adds nothing.
    (tmp_path / "again").mkdir()
    again = _publish_with_state(
        tmp_path / "again",
        monkeypatch,
        evidence(*published),
        evidence(record(B, day=1)),
    )
    assert again == published


def test_guard_flags_repo_lacking_a_resolved_local_record():
    from test_qualification_status import A, B, evidence, record

    from spark_sweep.qualification import resolved_missing

    unresolved = A.model_copy(update={"models_resolved": False, "model_digests": ()})
    local = evidence(record(A), record(B), record(unresolved))
    assert [r.identity for r in resolved_missing(local, evidence(record(A)))] == [B]
    assert resolved_missing(local, evidence(record(A), record(B))) == ()


def test_withdrawn_recipes_are_never_counted_as_missing(tmp_path):
    from test_qualification_status import KEY, A, evidence, record

    from spark_sweep.qualification import backfill

    seed_catalog(tmp_path)
    withdrawn = evidence(record(A)).model_copy(update={"recipe_key": "vonk-forge/gone"})
    result = backfill(
        [], tmp_path, {"vonk-forge/gone": withdrawn, KEY: evidence(record(A))}
    )
    assert result.unpublished == 1  # only the catalog recipe


def test_a_batch_that_never_reaches_backfill_leaves_no_stale_count(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    work.mkdir()
    (work / publisher.UNPUBLISHED_FILE).write_text('{"count": 5}')

    def command(cwd, *args, **kwargs):
        if args[:3] == ("gh", "pr", "list"):
            return '[{"number": 9}]'
        if args[:3] == ("gh", "pr", "view"):
            return '{"number": 9, "mergeStateStatus": "CLEAN", "statusCheckRollup": []}'
        return ""

    monkeypatch.setattr(publisher, "command", command)
    code = publisher.sync(tmp_path / "results.jsonl", work, tmp_path / "state.json")
    assert code == publisher.SweepEvidenceExit.WAITING
    assert not (work / publisher.UNPUBLISHED_FILE).exists()


def test_only_resolved_records_are_exported_and_unresolved_stay_local(
    tmp_path, monkeypatch
):
    from test_qualification_status import A, B, evidence, record

    unresolved = B.model_copy(update={"models_resolved": False, "model_digests": ()})
    published = _publish_with_state(
        tmp_path, monkeypatch, None, evidence(record(A), record(unresolved, day=1))
    )
    assert [r.identity for r in published] == [A]


def test_unreadable_state_prints_its_cause_and_returns_nothing(tmp_path, capsys):
    from spark_sweep.qualification import read_state_evidence

    broken = tmp_path / "state.json"
    broken.write_text("{not json")
    assert read_state_evidence(broken) == {}
    assert "sweep state unreadable: JSONDecodeError" in capsys.readouterr().out


def test_the_publisher_child_receives_the_sweep_state(tmp_path, monkeypatch):
    argv = []

    class Process:
        def __init__(self, args, **kwargs):
            argv.extend(args)

    monkeypatch.setattr(publisher.subprocess, "Popen", Process)
    state = tmp_path / "state.json"
    publisher.EvidencePublisher(
        tmp_path / "results.jsonl", tmp_path / "publisher", Path("worker"), state
    ).tick(0)
    assert argv[argv.index("--state") + 1] == str(state)


def test_the_sweep_wires_its_state_file_into_the_publisher(tmp_path):
    from test_qualification_status import evidence_sweep

    sweep, _, _ = evidence_sweep(tmp_path)
    assert sweep.evidence_publisher.state == sweep.state.path


def test_missing_records_show_in_the_publisher_message_and_alert_when_persistent(
    tmp_path, monkeypatch
):
    from test_qualification_status import evidence_sweep

    from spark_sweep.report import build_status, render_status

    sweep, _, _ = evidence_sweep(tmp_path)
    worker = sweep.evidence_publisher
    worker.directory.mkdir(parents=True, exist_ok=True)

    class Done:
        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    def batch(count, now):
        # The real child unlinks first and reports only if it reached backfill.
        report = worker.directory / publisher.UNPUBLISHED_FILE
        report.unlink(missing_ok=True)
        if count is not None:
            report.write_text(json.dumps({"count": count}))
        worker.process = Done()
        worker.next_attempt = 1e18
        worker.tick(now)

    batch(3, 0)
    status = build_status(sweep)
    assert "3 resolved sweep records were missing" in status["evidence_publication"]
    assert not any("evidence publisher" in a for a in status["alerts"])
    batch(3, 1)
    status = build_status(sweep)
    assert any("3 resolved sweep records still missing" in a for a in status["alerts"])
    assert "evidence publisher: 3 resolved" in render_status(status)
    batch(None, 2)  # WAITING/FAILING_CHECK exit: no new report, nothing changes
    assert worker.unpublished_batches == 2
    batch(0, 3)
    assert worker.alert is None


def test_the_same_red_check_twice_is_refused_alerts_and_new_evidence_still_flows(
    tmp_path, monkeypatch
):
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps(historical()) + "\n")
    work = tmp_path / "work"
    calls = []

    def command(cwd, *args, **kwargs):
        calls.append(args)
        if args[:3] == ("gh", "pr", "list"):
            return '[{"number":123}]'
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps(
                {
                    "number": 123,
                    "mergeStateStatus": "BLOCKED",
                    "headRefOid": "abc123",
                    "statusCheckRollup": [
                        {"name": "lint", "status": "COMPLETED", "conclusion": "FAILURE"}
                    ],
                }
            )
        if args[:2] == ("gh", "api"):
            return "0"  # main has not moved: the red PR would wait forever
        if args[:2] == ("git", "clone"):
            (Path(args[-1]) / ".git").mkdir()
            seed_catalog(Path(args[-1]))
        if args[:3] == ("git", "diff", "origin/main"):
            return "evidence diff"
        if args[:3] == ("git", "diff", "--cached"):
            return "qualification/evidence/vonk-forge/example.json"
        return ""

    monkeypatch.setattr(publisher, "command", command)
    first = publisher.sync(history, work)
    assert first == publisher.SweepEvidenceExit.FAILING_CHECK
    assert not any(c[:2] == ("git", "push") for c in calls)  # waits once
    publisher.sync(history, work)  # the same red twice: refused
    assert publisher.load_state(work) == (
        publisher.PublisherState.REFUSED,
        "abc123 lint",
    )
    assert any(c[:2] == ("git", "push") for c in calls)  # new evidence still flows

    class Done:
        def poll(self):
            return publisher.SweepEvidenceExit.SYNCED

        def wait(self, timeout):
            return 0

    worker = publisher.EvidencePublisher(history, work, Path("worker"))
    monkeypatch.setattr(worker, "process", Done())
    worker.next_attempt = 1e18
    worker.tick(0)
    assert worker.alert
    assert "refused, same failure twice: abc123 lint" in worker.alert
    # The push moves the head; a new, pending PR clears the refusal.
    assert (
        publisher.transition(
            publisher.PublisherState.REFUSED, red=False, repeated=False
        )
        == publisher.PublisherState.IDLE
    )
    assert (
        publisher.transition(publisher.PublisherState.REFUSED, red=True, repeated=False)
        == publisher.PublisherState.RED
    )


def test_unknown_state_file_self_heals_to_idle(tmp_path):
    (tmp_path / publisher.STATE_FILE).write_text('{"state":"bogus","signature":"x"}')
    assert publisher.load_state(tmp_path) == (publisher.PublisherState.IDLE, "")


def test_a_deterministic_command_failure_is_refused_not_a_crash(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            128, ["git", "push", "x"], stderr="fatal: bad object"
        )

    monkeypatch.setattr(publisher, "_batch", fail)
    work = tmp_path / "work"
    history = tmp_path / "results.jsonl"
    for expected in ("red", "refused"):
        assert (
            publisher.sync(history, work) == publisher.SweepEvidenceExit.FAILING_CHECK
        )
        assert publisher.load_state(work)[0] == expected


def test_a_transient_command_failure_retries_with_jitter_then_succeeds(monkeypatch):
    sleeps = []
    attempts = iter(
        [
            subprocess.CalledProcessError(1, ["gh"], stderr="HTTP 503"),
            subprocess.CompletedProcess(["gh"], 0, stdout="ok\n"),
        ]
    )

    def run(*args, **kwargs):
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(publisher.subprocess, "run", run)
    monkeypatch.setattr(publisher.time, "sleep", sleeps.append)
    assert publisher.command(Path("."), "gh") == "ok"
    assert len(sleeps) == 1 and 0 <= sleeps[0] <= 2
