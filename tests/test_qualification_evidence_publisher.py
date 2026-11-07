"""Fault injection proves evidence transport cannot hold hardware lanes."""

import json
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest
from test_qualification_status import historical

from spark_sweep import evidence_publisher as publisher


def pr_list(*numbers):
    """What `gh pr list` returns for PRs from this repository."""
    return json.dumps(
        [
            {
                "number": n,
                "isCrossRepository": False,
                "headRepositoryOwner": {"login": "CarstVaartjes"},
            }
            for n in numbers
        ]
    )


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
            return pr_list(123) if pushed else "[]"
        return ""

    monkeypatch.setattr(publisher, "command", command)
    first = publisher.sync(history, tmp_path / "work")
    assert first == publisher.SweepEvidenceExit.COMMAND_FAILED  # no in-process retry
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
            return "[]" if lists == 1 else pr_list(123)
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
            return pr_list(123)
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
            return "[]" if lists == 1 else pr_list(123)
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
            return pr_list(123)
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
        "evidence PR #123 has a failing check (check); waiting for main to move"
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
        "evidence PR has failing checks (); waiting for main to move (see publisher.log)"
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
            return "[]" if listed == 1 else pr_list(7)
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
            return pr_list(9)
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
            report.write_text(
                publisher.UnpublishedReport(count=count, head="h").model_dump_json()
            )
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


def _row(day):
    row = historical()
    row["recorded_at"] = f"2026-10-{day}T00:00:00+00:00"
    return row


def _evidence_json(rows):
    from test_qualification_status import KEY
    from vonk_forge_contracts import RecipeQualificationEvidence

    from spark_sweep.qualification import from_history

    records = tuple(r for r in map(from_history, rows) if r)
    return RecipeQualificationEvidence(
        recipe_key=KEY, records=records
    ).model_dump_json()


class World:
    """A fake GitHub for the real sequence: red on head A, main moves, re-plan push
    makes head B, B's checks run, B is red with the same names."""

    LINT: ClassVar = [{"name": "lint", "status": "COMPLETED", "conclusion": "FAILURE"}]

    def __init__(self, tmp_path, monkeypatch, *, close_fails=False, branch=None):
        from spark_sweep.qualification import read_evidence

        self.refused_row, self.new_row = _row(21), _row(22)
        self.history = tmp_path / "results.jsonl"
        self.history.write_text(
            "\n".join(json.dumps(r) for r in (self.refused_row, self.new_row)) + "\n"
        )
        self.work = tmp_path / "work"
        self.calls = []
        self.open_prs = [123]
        self.head = "A"
        self.rollup = self.LINT
        self.behind = 0
        self.created = []
        self.close_fails = close_fails
        self.branch = branch or _evidence_json([self.refused_row])
        self.pushes = 0
        self.pushed = []

        def command(cwd, *args, **kwargs):
            from test_qualification_status import KEY

            self.calls.append(args)
            if args[:3] == ("gh", "pr", "list"):
                return json.dumps(
                    [
                        {
                            "number": n,
                            "isCrossRepository": False,
                            "headRepositoryOwner": {"login": "CarstVaartjes"},
                        }
                        for n in self.open_prs
                    ]
                )
            if args[:3] == ("gh", "pr", "view"):
                return json.dumps(
                    {
                        "number": 123,
                        "mergeStateStatus": "BLOCKED",
                        "headRefOid": self.head,
                        "statusCheckRollup": self.rollup,
                    }
                )
            if args[:2] == ("gh", "api"):
                return str(self.behind)
            if args[:3] == ("gh", "pr", "close"):
                if self.close_fails:
                    raise subprocess.CalledProcessError(1, args, stderr="boom")
                self.open_prs.clear()
            if args[:3] == ("gh", "pr", "create"):
                self.open_prs.append(124)
                self.created.append(read_evidence(cwd, KEY).records)
            if args[:2] == ("git", "clone"):
                (Path(args[-1]) / ".git").mkdir()
                seed_catalog(Path(args[-1]))
            if args[:2] == ("git", "ls-remote"):
                return "sha refs/heads/x" if self.open_prs else ""
            if args[:2] == ("git", "push"):
                self.pushed.append(read_evidence(cwd, KEY).records)
                self.head = "B"
                self.pushes += 1
            if args[:2] == ("git", "show"):
                if args[2].startswith("origin/qualification"):
                    return self.branch
                raise subprocess.CalledProcessError(128, args)  # not on main
            if args[:2] == ("git", "diff") and "--name-only" in args:
                return "qualification/evidence/vonk-forge/example.json"
            if args[:3] == ("git", "diff", "origin/main"):
                return "evidence diff"
            if args[:3] == ("git", "diff", "--cached"):
                return "qualification/evidence/vonk-forge/example.json"
            return ""

        monkeypatch.setattr(publisher, "command", command)
        # the merge of main into the evidence branch is a plain subprocess call
        monkeypatch.setattr(
            publisher.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "", ""),
        )

    def tick(self):
        return publisher.sync(self.history, self.work)

    def closed(self):
        return [c for c in self.calls if c[:3] == ("gh", "pr", "close")]


def test_the_real_red_sequence_ends_refused_with_a_fresh_pr_and_an_alert(
    tmp_path, monkeypatch
):
    exits = publisher.SweepEvidenceExit
    world = World(tmp_path, monkeypatch)
    # 1. red on head A while main has not moved: RED, waits
    assert world.tick() == exits.FAILING_CHECK
    assert world.pushes == 0 and not world.closed()
    assert publisher.load_state(world.work).state == publisher.PublisherState.RED
    # 2. the same head on the next tick is the same run: still RED, still waiting
    assert world.tick() == exits.FAILING_CHECK
    assert publisher.load_state(world.work).state == publisher.PublisherState.RED
    # 3. main moves: the re-plan pushes head B
    world.behind = 3
    world.tick()
    assert world.pushes == 1 and world.head == "B" and not world.closed()
    # 4. B's checks are still running: nothing observed, the record is unchanged
    before = publisher.load_state(world.work)
    world.rollup = [{"status": "IN_PROGRESS", "conclusion": ""}]
    world.behind = 0
    assert world.tick() == exits.WAITING
    assert publisher.load_state(world.work) == before
    # 5. B is red with the same names: a second run, so REFUSED
    world.rollup = world.LINT
    assert world.tick() == exits.REFUSED
    close = world.closed()[0]
    assert close[3] == "123" and "lint" in close[-1] and "--delete-branch" in close
    assert world.open_prs == [124]
    (records,) = world.created
    assert [r.tested_at.day for r in records] == [22]  # the refused PR's record is out
    record = publisher.load_state(world.work)
    assert record.state == publisher.PublisherState.REFUSED
    assert record.signature == "lint" and len(record.refused) == 1

    class Done:
        def poll(self):
            return exits.REFUSED

        def wait(self, timeout):
            return 0

    worker = publisher.EvidencePublisher(world.history, world.work, Path("worker"))
    monkeypatch.setattr(worker, "process", Done())
    worker.next_attempt = 1e18
    worker.tick(0)
    assert worker.alert and "1 records refused" in worker.alert
    assert "refused (red twice: lint)" in worker.message


def test_a_slow_red_is_not_reset_by_a_pending_tick(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    world.tick()  # red on A
    world.rollup = [{"status": "IN_PROGRESS", "conclusion": ""}]
    for _ in range(3):
        world.tick()
        assert publisher.load_state(world.work).state == publisher.PublisherState.RED


def test_refused_ids_are_recorded_only_after_the_close_succeeded(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch, close_fails=True)
    world.tick()  # red on A
    world.head = "B"  # the next red run
    assert world.tick() == publisher.SweepEvidenceExit.COMMAND_FAILED
    record = publisher.load_state(world.work)
    assert record.refused == () and record.state == publisher.PublisherState.RED
    assert world.open_prs == [123] and not world.created
    world.close_fails = False  # next tick retries the close from the persisted state
    assert world.tick() == publisher.SweepEvidenceExit.REFUSED
    assert world.open_prs == [124]
    assert len(publisher.load_state(world.work).refused) == 1


def test_malformed_branch_evidence_is_a_command_failure_not_an_escape(
    tmp_path, monkeypatch, capsys
):
    world = World(tmp_path, monkeypatch, branch='{"recipe_key": 7}')
    world.tick()
    world.head = "B"
    capsys.readouterr()
    assert world.tick() == publisher.SweepEvidenceExit.COMMAND_FAILED
    out = capsys.readouterr().out  # the real error reaches publisher.log
    assert "ValidationError" in out and "recipe_key" in out
    record = publisher.load_state(world.work)
    assert record.command and record.signature == "validate evidence exit invalid"
    assert not world.closed()


def test_refused_records_stay_out_until_their_content_changes(tmp_path, monkeypatch):
    from test_qualification_status import KEY

    from spark_sweep.qualification import read_evidence

    old, newer = _row(21), _row(23)
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps(old) + "\n")
    work = tmp_path / "work"
    work.mkdir()
    from spark_sweep.qualification import from_history

    record = from_history(old)
    assert record
    refused = publisher.record_id(KEY, record)
    (work / publisher.STATE_FILE).write_text(
        publisher.PublisherRecord(
            refused=(refused,), refused_because="lint"
        ).model_dump_json()
    )
    seen = []

    def command(cwd, *args, **kwargs):
        if args[:3] == ("gh", "pr", "list"):
            return "[]"
        if args[:2] == ("git", "clone"):
            (Path(args[-1]) / ".git").mkdir()
            seed_catalog(Path(args[-1]))
        if args[:3] == ("git", "diff", "origin/main"):
            seen.append(read_evidence(cwd, KEY).records)
            return ""
        return ""

    monkeypatch.setattr(publisher, "command", command)
    publisher.sync(history, work)
    assert seen[-1] == ()  # the refused record is left out
    assert publisher.load_state(work).refused == (refused,)
    history.write_text(json.dumps(old) + "\n" + json.dumps(newer) + "\n")
    publisher.sync(history, work)
    assert [r.tested_at.day for r in seen[-1]] == [23]  # newer evidence supersedes it


def test_a_failing_command_is_keyed_on_argv_and_exit_code_not_text(
    tmp_path, monkeypatch
):
    texts = iter(["fatal: first wording", "fatal: other wording"])

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            128, ["git", "push", "x"], stderr=next(texts)
        )

    monkeypatch.setattr(publisher, "_batch", fail)
    work = tmp_path / "work"
    history = tmp_path / "results.jsonl"
    for expected in ("red", "refused"):
        code = publisher.sync(history, work)
        assert code == publisher.SweepEvidenceExit.COMMAND_FAILED
        record = publisher.load_state(work)
        assert (record.state, record.signature) == (expected, "git push exit 128")
        assert record.command


def test_success_clears_the_state_and_unknown_state_self_heals(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / publisher.STATE_FILE).write_text('{"state":"bogus"}')
    assert publisher.load_state(work) == publisher.PublisherRecord()
    monkeypatch.setattr(
        publisher, "_batch", lambda *a: publisher.SweepEvidenceExit.SYNCED
    )
    publisher.sync(tmp_path / "results.jsonl", work)
    assert publisher.load_state(work).state == publisher.PublisherState.IDLE


def test_refused_needs_two_runs_and_is_reachable_only_after_red():
    idle = publisher.PublisherRecord()
    red = publisher.transition(idle, signature="lint", head="h1")
    assert red.state == publisher.PublisherState.RED
    same_run = publisher.transition(red, signature="lint", head="h1")
    assert same_run.state == publisher.PublisherState.RED  # one run seen twice
    again = publisher.transition(red, signature="lint", head="h2")
    assert again.state == publisher.PublisherState.REFUSED
    other = publisher.transition(red, signature="other", head="h2")
    assert other.state == publisher.PublisherState.RED
    assert publisher.transition(red, signature=None).state == (
        publisher.PublisherState.IDLE
    )
    # a refused batch does not count as the first of two reds for the next PR
    fresh = publisher.transition(again, signature="lint", head="h3")
    assert fresh.state == publisher.PublisherState.RED


def test_the_same_red_run_seen_on_two_ticks_is_not_refused(tmp_path, monkeypatch):
    history, calls = _open_pr(tmp_path, monkeypatch, RED, 0)
    work = tmp_path / "work"
    for _ in range(3):
        assert (
            publisher.sync(history, work) == publisher.SweepEvidenceExit.FAILING_CHECK
        )
    assert publisher.load_state(work).state == publisher.PublisherState.RED
    assert not any(c[:3] == ("gh", "pr", "close") for c in calls)


def test_refused_because_is_set_only_when_the_refused_set_grows():
    first = publisher.transition(
        publisher.PublisherRecord(), signature="lint", refused=("a#1",)
    )
    assert first.refused_because == "lint"
    later = publisher.transition(first, signature="other", refused=("a#1",))
    assert later.refused_because == "lint"
    grown = publisher.transition(later, signature="other", refused=("a#1", "b#2"))
    assert grown.refused_because == "other"
    assert publisher.transition(grown, signature=None, refused=()).refused_because == ""


def test_refused_records_are_released_when_main_moves(tmp_path, monkeypatch):
    from test_qualification_status import KEY

    from spark_sweep.qualification import from_history, read_evidence

    row = _row(21)
    record = from_history(row)
    assert record
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps(row) + "\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / publisher.STATE_FILE).write_text(
        publisher.PublisherRecord(
            refused=(publisher.record_id(KEY, record),),
            refused_because="lint",
            main="old",
        ).model_dump_json()
    )
    main = {"sha": "old"}
    seen = []

    def command(cwd, *args, **kwargs):
        if args[:3] == ("gh", "pr", "list"):
            return "[]"
        if args[:2] == ("git", "clone"):
            (Path(args[-1]) / ".git").mkdir()
            seed_catalog(Path(args[-1]))
        if args[:2] == ("git", "rev-parse"):
            return main["sha"]
        if args[:3] == ("git", "diff", "origin/main"):
            seen.append(read_evidence(cwd, KEY).records)
        return ""

    monkeypatch.setattr(publisher, "command", command)
    publisher.sync(history, work)
    assert seen[-1] == () and publisher.load_state(work).refused  # still refused
    main["sha"] = "new"  # main moved: the red may be fixed there
    publisher.sync(history, work)
    assert len(seen[-1]) == 1
    released = publisher.load_state(work)
    assert released.refused == () and released.main == ""


def test_dropping_refused_records_reads_only_the_files_it_names(tmp_path, monkeypatch):
    from test_qualification_status import KEY
    from vonk_forge_contracts import RecipeQualificationEvidence

    from spark_sweep.qualification import from_history, write_evidence

    record = from_history(historical())
    assert record
    for n in range(300):
        write_evidence(
            tmp_path,
            RecipeQualificationEvidence(
                recipe_key=f"vonk-forge/r{n}", records=(record,)
            ),
        )
    reads = []
    real = publisher.read_evidence
    monkeypatch.setattr(
        publisher,
        "read_evidence",
        lambda root, key: reads.append(key) or real(root, key),
    )
    refused = {publisher.record_id(f"vonk-forge/r{n}", record) for n in (3, 250)}
    found = publisher._drop_refused(tmp_path, refused)
    assert sorted(reads) == ["vonk-forge/r250", "vonk-forge/r3"]
    assert len(found) == 2
    assert not (tmp_path / "qualification/evidence/vonk-forge/r3.json").exists()
    assert (tmp_path / "qualification/evidence/vonk-forge/r4.json").exists()
    assert KEY not in reads


#: Every tool that opens pull requests, with what happens when its PR cannot land.
#: "refused": the batch stops waiting, leaves the blocked content out and alerts.
#: "follow-up": no refused outcome yet, with the reason it is not part of this change.
PR_OPENING_TOOLS = {
    "spark_sweep/evidence_publisher.py": (
        "refused",
        "PublisherState.REFUSED: PR closed, records left out until main moves, alert",
    ),
    "tools/refresh-upstream": (
        "follow-up",
        (
            "failed/cancelled refresh PRs wait for upstream or main to change; no alert "
            "and no fork-PR check on its --head / headRefName selection"
        ),
    ),
}
#: Named in the audit but opens no PR; kept so a change that makes it open one fails here.
OPENS_NO_PR = {
    "tools/sync-release-assets": "uploads release assets with gh release upload",
    "tools/backfill-qualification-evidence": "writes evidence files in a checkout",
}


def test_every_tool_that_opens_pull_requests_declares_its_refused_outcome():
    import re

    root = Path(__file__).resolve().parents[1]
    opens = re.compile(r'"pr",\s*"create"|\bgh\s+pr\s+create\b')
    found = {
        str(path.relative_to(root))
        for path in [
            *(root / "tools").glob("*"),
            *(root / "spark_sweep").glob("*.py"),
            *(root / ".github/workflows").glob("*.yml"),
        ]
        if path.is_file() and opens.search(path.read_text(errors="ignore"))
    }
    assert found == set(PR_OPENING_TOOLS), (
        "a tool opens pull requests without a declared refused outcome: "
        f"{sorted(found ^ set(PR_OPENING_TOOLS))}"
    )
    for path, (kind, reason) in PR_OPENING_TOOLS.items():
        assert kind in ("refused", "follow-up") and reason, path
    for path in OPENS_NO_PR:
        assert path not in found


def test_a_failed_command_logs_its_real_stderr(tmp_path, monkeypatch, capsys):
    world = World(tmp_path, monkeypatch, close_fails=True)
    world.tick()
    world.head = "B"
    capsys.readouterr()
    world.tick()
    out = capsys.readouterr().out
    assert "CalledProcessError" in out and "stderr: boom" in out


def test_an_empty_rollup_on_a_new_head_is_pending_not_settled(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    world.tick()  # red on A
    before = publisher.load_state(world.work)
    world.head, world.rollup = "B", []  # just pushed: CI has not registered yet
    assert world.tick() == publisher.SweepEvidenceExit.WAITING
    assert publisher.load_state(world.work) == before


def test_a_fork_pr_on_the_same_branch_is_never_merged_closed_or_counted(
    tmp_path, monkeypatch
):
    fork = {
        "number": 666,
        "isCrossRepository": True,
        "headRepositoryOwner": {"login": "someone-else"},
    }
    spoofed_owner = {**fork, "number": 667, "isCrossRepository": False}
    calls = []

    def command(cwd, *args, **kwargs):
        calls.append(args)
        if args[:3] == ("gh", "pr", "list"):
            return json.dumps([fork, spoofed_owner, *json.loads(pr_list(123))])
        if args[:3] == ("gh", "pr", "view"):
            assert args[3] == "123"
            return json.dumps(
                {
                    "number": 123,
                    "mergeStateStatus": "BLOCKED",
                    "statusCheckRollup": PENDING,
                }
            )
        return ""

    monkeypatch.setattr(publisher, "command", command)
    history = tmp_path / "results.jsonl"
    history.write_text("")
    assert (
        publisher.sync(history, tmp_path / "work")
        == publisher.SweepEvidenceExit.WAITING
    )  # one own PR plus two foreign ones is not "multiple PRs"
    touched = [c for c in calls if c[:2] == ("gh", "pr") and c[2] != "list"]
    assert touched and all(c[3] == "123" for c in touched)
    assert any(c[:3] == ("gh", "pr", "merge") for c in touched)
    assert not any("666" in c or "667" in c for c in calls)
    assert any("isCrossRepository" in a for c in calls for a in c)


def test_a_final_batch_updates_an_open_pr_that_normal_ticks_leave_waiting(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    world.rollup = [{"status": "IN_PROGRESS", "conclusion": ""}]
    assert world.tick() == publisher.SweepEvidenceExit.WAITING
    assert world.pushes == 0  # the open PR does not carry the newest results yet
    code = publisher.sync(world.history, world.work, final=True)
    assert code == publisher.SweepEvidenceExit.SYNCED
    assert world.pushes == 1 and world.open_prs == [123]
    (records,) = world.pushed  # the newest row is in the pushed branch content
    assert {r.tested_at.day for r in records} == {21, 22}
    assert any(
        c[:3] == ("gh", "pr", "merge") for c in world.calls
    )  # requested, not awaited


def test_the_unpublished_count_is_stale_once_the_history_moves_past_it(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    unknown = publisher.UnpublishedStatus()
    assert publisher.unpublished_report(world.work, world.history) == unknown
    publisher.sync(world.history, world.work, final=True)
    counted = publisher.unpublished_report(world.work, world.history)
    assert counted.state == publisher.UnpublishedState.COUNTED
    assert isinstance(counted.count, int)
    with world.history.open("a") as stream:  # a pass finishes after the count
        stream.write(json.dumps(_row(23)) + "\n")
    stale = publisher.unpublished_report(world.work, world.history)
    assert stale == publisher.UnpublishedStatus(state=publisher.UnpublishedState.STALE)
    # A report without a head (any other shape) is unknown, never a trusted zero.
    (world.work / publisher.UNPUBLISHED_FILE).write_text('{"count": 0}')
    assert publisher.unpublished_report(world.work, world.history) == unknown


def test_the_status_shows_stale_instead_of_a_count(tmp_path):
    from test_qualification_status import evidence_sweep

    from spark_sweep.report import build_status, render_status

    sweep, _, _ = evidence_sweep(tmp_path)
    worker = sweep.evidence_publisher
    worker.directory.mkdir(parents=True, exist_ok=True)
    worker.history.write_text(json.dumps(_row(21)) + "\n")
    report = publisher.UnpublishedReport(
        count=0, head=publisher.history_head(worker.history)
    )
    (worker.directory / publisher.UNPUBLISHED_FILE).write_text(report.model_dump_json())
    assert build_status(sweep)["evidence_unpublished"] == {
        "state": "counted",
        "count": 0,
    }
    with worker.history.open("a") as stream:
        stream.write(json.dumps(_row(22)) + "\n")
    status = build_status(sweep)
    assert status["evidence_unpublished"] == {"state": "stale", "count": None}
    assert "evidence unpublished: stale" in render_status(status)


class _Child:
    """A publisher child; `code` is its exit, None means it hangs until killed."""

    started: ClassVar[list] = []
    code: ClassVar[int | None] = 0

    def __init__(self, args, **kwargs):
        self.args, self.killed = args, False
        self.exit = type(self).code
        _Child.started.append(self)

    def poll(self):
        return None

    def wait(self, timeout=None):
        if self.killed:
            return -9
        if self.exit is None:
            if timeout is None:
                raise AssertionError("an unbounded wait on a hung child")
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.exit

    def kill(self):
        self.killed = True


def test_flush_kills_a_running_child_and_runs_one_bounded_final_batch(
    tmp_path, monkeypatch
):
    _Child.started.clear()
    monkeypatch.setattr(_Child, "code", 0)
    monkeypatch.setattr(publisher.subprocess, "Popen", _Child)
    worker = publisher.EvidencePublisher(
        tmp_path / "results.jsonl", tmp_path / "publisher", Path("worker")
    )
    worker.tick(0)
    assert len(_Child.started) == 1 and "--final" not in _Child.started[0].args
    assert worker.flush(1) == publisher.SweepEvidenceExit.SYNCED
    assert _Child.started[0].killed  # its history was older: not awaited
    assert len(_Child.started) == 2 and "--final" in _Child.started[1].args
    assert worker.process is None and worker.message == "evidence sync complete"


def test_a_hung_final_batch_is_killed_and_the_results_stay_for_the_next_run(
    tmp_path, monkeypatch, capsys
):
    _Child.started.clear()
    monkeypatch.setattr(_Child, "code", None)
    monkeypatch.setattr(publisher.subprocess, "Popen", _Child)
    history = tmp_path / "results.jsonl"
    history.write_text(json.dumps(_row(21)) + "\n")
    worker = publisher.EvidencePublisher(history, tmp_path / "publisher", Path("w"))
    worker.tick(0)  # a stale child that also hangs
    assert worker.flush(1) is None
    assert all(child.killed for child in _Child.started) and worker.process is None
    assert "exceeded" in worker.message
    assert "stop again to skip" in capsys.readouterr().out
    # nothing was counted, so a reader sees the gap as unknown, never as zero
    assert worker.unpublished_report == publisher.UnpublishedStatus()


def test_flush_reports_a_held_pr_and_survives_a_second_stop(tmp_path, monkeypatch):
    _Child.started.clear()
    monkeypatch.setattr(_Child, "code", 11)
    monkeypatch.setattr(publisher.subprocess, "Popen", _Child)
    worker = publisher.EvidencePublisher(
        tmp_path / "results.jsonl", tmp_path / "publisher", Path("worker")
    )
    assert worker.flush(0) == publisher.SweepEvidenceExit.FAILING_CHECK
    assert "failing checks" in worker.message

    def stopped(self, timeout=None):
        if not self.killed:
            raise KeyboardInterrupt
        return -9

    monkeypatch.setattr(_Child, "wait", stopped)
    assert worker.flush(1) is None
    assert _Child.started[-1].killed and "interrupted" in worker.message


def test_a_final_batch_on_a_red_pr_that_pushes_nothing_reports_failing_check(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    world.branch = _evidence_json([world.refused_row, world.new_row])
    fake = publisher.command

    def command(cwd, *args, **kwargs):
        if args[:3] == ("git", "diff", "--cached"):
            return ""  # the clone already holds everything: nothing to commit
        return fake(cwd, *args, **kwargs)

    monkeypatch.setattr(publisher, "command", command)
    code = publisher.sync(world.history, world.work, final=True)
    assert code == publisher.SweepEvidenceExit.FAILING_CHECK
    assert world.pushes == 0


def _flushing_sweep(tmp_path, monkeypatch, publish=True):
    from test_qualification_status import evidence_sweep

    sweep, _, _ = evidence_sweep(tmp_path)
    sweep.cfg.publish_evidence = publish
    sweep.qualification_evidence = {"k": object()}
    flushed = []
    monkeypatch.setattr(sweep.evidence_publisher, "flush", flushed.append)
    return sweep, flushed


@pytest.mark.parametrize("how", ["returns", "raises"])
def test_the_sweep_flushes_evidence_whatever_ends_it(tmp_path, monkeypatch, how):
    from spark_sweep import run as run_module

    sweep, flushed = _flushing_sweep(tmp_path, monkeypatch)

    def ended():
        if how == "raises":
            raise RuntimeError("boom")
        return 130  # a SIGTERM shutdown returns 130 after finish()

    monkeypatch.setattr(sweep, "_run", ended)
    monkeypatch.setattr(run_module, "_install_signal_handlers", dict)
    if how == "raises":
        with pytest.raises(RuntimeError):
            sweep.run()
    else:
        assert sweep.run() == 130
    assert len(flushed) == 1


def test_no_flush_without_publication_or_evidence(tmp_path, monkeypatch):
    sweep, flushed = _flushing_sweep(tmp_path, monkeypatch, publish=False)
    sweep._flush_evidence()
    sweep.cfg.publish_evidence = True
    sweep.qualification_evidence = {}
    sweep._flush_evidence()
    assert flushed == []


def test_a_late_stop_signal_cannot_change_the_exit_or_skip_the_release(
    tmp_path, monkeypatch
):
    from spark_sweep import run as run_module

    sweep, _ = _flushing_sweep(tmp_path, monkeypatch)
    restored = []

    def late_stop(now):
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep.evidence_publisher, "flush", late_stop)
    monkeypatch.setattr(sweep, "_run", lambda: 0)
    monkeypatch.setattr(run_module, "_install_signal_handlers", dict)
    monkeypatch.setattr(
        run_module, "_restore_signal_handlers", lambda previous: restored.append(1)
    )
    assert sweep.run() == 0 and restored == [1]
    status = json.loads((sweep.cfg.state_dir / "status.json").read_text())
    assert (
        "evidence_unpublished" in status
    )  # the last status was written after the flush


def test_every_tick_driven_child_process_worker_is_flushed_on_exit():
    """Net for the class: a worker that starts child processes from tick() must be
    flushed by Sweep.run(); only the evidence publisher strands results today."""
    import ast

    root = Path(publisher.__file__).parent
    workers = set()
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            source = ast.unparse(node)
            if "tick" in methods and "subprocess.Popen" in source:
                assert "flush" in methods, (
                    f"{node.name} starts children but has no flush"
                )
                workers.add(node.name)
    assert workers == {"EvidencePublisher"}
    run_source = (root / "run.py").read_text()
    assert "self.evidence_publisher.flush(" in run_source
    assert (
        "self._flush_evidence()"
        in run_source.split("def run(self)")[1].split("def _flush_evidence")[0]
    )
