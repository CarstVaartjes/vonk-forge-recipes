"""Class guards: published vocabulary, bounded Controller waits, evidence and untyped access.

The sweep follows the Controller's verdicts (``spark_sweep/verdict.py``). These guards are
AST-based and whole-tree over ``spark_sweep`` (it is small):

* all wait budgets go through one primitive; own-work stalls follow progress;
* a recipe's attempt counter moves only after ``disposition(...)`` decided the outcome;
* every lane exit writes a results row, through the one writer ``record_result``.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from spark_sweep import verdict
from spark_sweep.lifecycle import ACTIVE, STATES, TERMINAL, WAITING

ROOT = Path(__file__).resolve().parents[1]
SWEEP = [*sorted((ROOT / "spark_sweep").glob("*.py")), ROOT / "tools/sweep-recipes"]


def test_fake_sweeps_cannot_launch_the_live_evidence_publisher(tmp_path, monkeypatch):
    from spark_sweep.evidence_publisher import EvidencePublisher

    def subprocess_forbidden(*args, **kwargs):
        raise AssertionError("a publisher subprocess escaped test isolation")

    monkeypatch.setattr(
        "spark_sweep.evidence_publisher.subprocess.Popen", subprocess_forbidden
    )
    directory = tmp_path / "publisher"
    publisher = EvidencePublisher(tmp_path / "results.jsonl", directory, Path("worker"))
    with pytest.raises(pytest.fail.Exception, match="disable evidence publication"):
        publisher.tick(0)
    assert not directory.exists()


def _functions(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _name(node: ast.AST) -> str | None:
    """The name a call, attribute, subscript key or variable refers to."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return str(node.slice.value)
    return None


def _calls(node: ast.AST, names: frozenset[str] | set[str]) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and _name(item.func) in names
    ]


# -- P1: a wait is not a failure; only a stall ends progressing work ------------------

#: Calls that end a lane's work or put a verdict on it.
ENDING = frozenset(
    {
        "_cancel_load",
        "_end_load",
        "_fail_slot",
        "_record_failure",
        "_retire",
        "_stalled",
        "_timeout",
        "cancel",
        "requeue",
    }
)


def unbounded_whiles(source: str) -> list[int]:
    """A loop waits on progress: it asks ``progress_verdict`` or its bound advances in its body."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.While):
            continue
        if _calls(node, {"progress_verdict"}):
            continue
        bounds = {
            name
            for item in ast.walk(node.test)
            if isinstance(item, ast.Compare)
            for part in ast.walk(item)
            if (name := _name(part)) is not None
        }
        advancing = any(
            _name(target) in bounds
            for statement in node.body
            for item in ast.walk(statement)
            if isinstance(item, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            for target in (
                item.targets if isinstance(item, ast.Assign) else [item.target]
            )
        )
        if not advancing:
            bad.append(node.lineno)
    return bad


def _is_clock(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "now":
        return True
    return isinstance(node, ast.Call) and _name(node.func) in {
        "now",
        "clock",
        "monotonic",
        "time",
    }


def fixed_caps_ending_work(source: str) -> list[int]:
    """A clock compared with a fixed offset (``now - since > CAP``) never ends work.

    Waits use the shared primitive (``progress_verdict``); a fixed cap may pace a poll or a
    backoff, but a branch that ends a lane, records a verdict or raises on it fails CI.
    """
    bad = []
    for function in _functions(ast.parse(source)):
        if _calls(function, {"progress_verdict"}):
            continue
        timed: set[str] = set()  # names assigned from the clock
        capped: set[str] = set()  # names assigned from an offset (a cap)
        for item in ast.walk(function):
            if isinstance(item, ast.Assign):
                names = {_name(t) for t in item.targets} - {None}
                if any(_is_clock(part) for part in ast.walk(item.value)):
                    timed |= names  # type: ignore[arg-type]
                if any(isinstance(part, ast.BinOp) for part in ast.walk(item.value)):
                    capped |= names  # type: ignore[arg-type]
        for item in ast.walk(function):
            if not isinstance(item, ast.If):
                continue
            compares = [c for c in ast.walk(item.test) if isinstance(c, ast.Compare)]
            for compare in compares:
                if _calls(compare, {"last_progress"}):
                    continue  # semantic output age, not a total stream duration
                parts = list(ast.walk(compare))
                clock = any(
                    _is_clock(p) or (isinstance(p, ast.Name) and p.id in timed)
                    for p in parts
                )
                cap = any(
                    isinstance(p, ast.BinOp)
                    or (isinstance(p, ast.Name) and p.id in capped)
                    for p in parts
                )
                ends = any(
                    _calls(statement, ENDING)
                    or any(
                        isinstance(s, ast.Raise)
                        and not (
                            isinstance(s.exc, ast.Call)
                            and _name(s.exc.func) == "EvidencePublicationTimeout"
                        )
                        for s in ast.walk(statement)
                    )
                    for statement in item.body
                )
                if clock and cap and ends:
                    bad.append(item.lineno)
                    break
    return bad


def test_wait_guard_accepts_progress_and_rejects_fixed_caps() -> None:
    assert unbounded_whiles("while True:\n    poll()") == [1]
    assert unbounded_whiles(
        "deadline = now() + 60\nwhile now() < deadline:\n    poll()"
    ) == [2]
    # A stall window: the bound advances when the work moves.
    assert unbounded_whiles("while now() < deadline:\n    deadline = now() + 60") == []
    assert (
        unbounded_whiles(
            "while progress_verdict(s, w, window=1, now=now()) is not STALLED:\n"
            "    poll()"
        )
        == []
    )
    capped = (
        "def f(self, slot, now):\n"
        "    if now - slot['since'] > self.cfg.blocked_seconds:\n"
        "        self._end_load(slot, 'x')\n"
    )
    assert fixed_caps_ending_work(capped) == [2]
    derived = (
        "def f(self, slot, now):\n"
        "    stalled = now - slot['at']\n"
        "    if stalled > self.cfg.copy_stall_seconds:\n"
        "        self._timeout(slot)\n"
    )
    assert fixed_caps_ending_work(derived) == [3]
    paced = (
        "def f(self, now):\n"
        "    if now - self.owner_at >= self.cfg.owner_poll_seconds:\n"
        "        self.check()\n"
    )
    assert fixed_caps_ending_work(paced) == []
    judged = (
        "def f(self, slot, now):\n"
        "    progress = progress_verdict(signal, slot, window=1, now=now)\n"
        "    if now - slot['at'] > 5:\n"
        "        self._timeout(slot)\n"
    )
    assert fixed_caps_ending_work(judged) == []
    # The optional repository publisher has its own bounded transport. Its typed
    # timeout must not end a hardware lane or spend a recipe attempt.
    transport = (
        "def command(now, started):\n"
        "    if now - started > 240:\n"
        "        raise EvidencePublicationTimeout('retry metadata transport')\n"
    )
    assert fixed_caps_ending_work(transport) == []
    assert fixed_caps_ending_work(
        transport.replace("EvidencePublicationTimeout", "TimeoutError")
    ) == [2]
    assert fixed_caps_ending_work(
        transport.replace(
            "raise EvidencePublicationTimeout",
            "self._end_load(slot)\n        raise EvidencePublicationTimeout",
        )
    ) == [2]


@pytest.mark.parametrize("path", SWEEP, ids=lambda p: p.name)
def test_every_sweep_wait_follows_progress_not_a_fixed_cap(path: Path) -> None:
    source = path.read_text()
    assert unbounded_whiles(source) == [], str(path)
    assert fixed_caps_ending_work(source) == [], str(path)


# -- P2: attempts are spent only on OBSERVE outcomes -----------------------------------

#: Counters that decide a recipe's fate.
FATE_COUNTERS = frozenset(
    {
        "attempts",
        "observe_failures",
        "review_defers",
        "review_freeing",
        "unattributed_requeues",
    }
)
#: Counters of the same name that pace a retry or re-plan and never decide a recipe's fate.
CADENCE = {
    (
        "_leave_fleet",
        "attempts",
    ): "backoff of a durable final fleet intent; never an exhaustion gate",
    ("cleanup_finished", "attempts"): "backoff exponent of an assignment removal retry",
    (
        "_defer_cleanup",
        "attempts",
    ): "backoff exponent of an unreadable cleanup receipt or a present target",
    (
        "_take_over",
        "attempts",
    ): "re-plan trigger: place on free online lanes; ends nothing",
    (
        "_record_pass",
        "attempts",
    ): "a pass is a definite outcome; no budget is consulted",
}


def _increments(function: ast.AST) -> list[tuple[int, str]]:
    found = []
    for item in ast.walk(function):
        if isinstance(item, ast.AugAssign) and isinstance(item.op, ast.Add):
            if (name := _name(item.target)) in FATE_COUNTERS:
                found.append((item.lineno, name))
        elif isinstance(item, ast.Assign) and any(
            isinstance(part, ast.BinOp) and isinstance(part.op, ast.Add)
            for part in ast.walk(item.value)
        ):
            for target in item.targets:
                if (name := _name(target)) in FATE_COUNTERS:
                    found.append((item.lineno, name))
        elif isinstance(item, ast.Call):
            for keyword in item.keywords:
                if keyword.arg in FATE_COUNTERS and any(
                    isinstance(part, ast.BinOp) and isinstance(part.op, ast.Add)
                    for part in ast.walk(keyword.value)
                ):
                    found.append((item.lineno, keyword.arg))
    return found


def counted_before_disposition(source: str) -> list[tuple[str, int, str]]:
    bad = []
    for function in _functions(ast.parse(source)):
        decided = [call.lineno for call in _calls(function, {"disposition"})]
        for line, name in _increments(function):
            if (function.name, name) in CADENCE:
                continue
            if not any(at <= line for at in decided):
                bad.append((function.name, line, name))
    return bad


def test_attempt_guard_requires_a_disposition_first() -> None:
    early = (
        "def f(self, entry, failure):\n"
        "    entry['attempts'] = entry.get('attempts', 0) + 1\n"
        "    if failure.transient:\n"
        "        return\n"
    )
    assert counted_before_disposition(early) == [("f", 2, "attempts")]
    decided = (
        "def f(self, entry, failure):\n"
        "    outcome = verdict.disposition(failure)\n"
        "    if outcome is WAIT:\n"
        "        return\n"
        "    entry['attempts'] = entry.get('attempts', 0) + 1\n"
    )
    assert counted_before_disposition(decided) == []
    assert counted_before_disposition("def g(e):\n    e.update(attempts=e['n'] + 1)\n")


@pytest.mark.parametrize("path", SWEEP, ids=lambda p: p.name)
def test_attempts_are_counted_only_after_a_disposition(path: Path) -> None:
    assert counted_before_disposition(path.read_text()) == [], str(path)


def test_cadence_exemptions_name_real_functions() -> None:
    names = {
        function.name
        for path in SWEEP
        for function in _functions(ast.parse(path.read_text()))
    }
    assert {function for function, _ in CADENCE} <= names


# -- P6: evidence is never lost --------------------------------------------------------

#: The lane exits: each writes its own row.
LANE_EXITS = frozenset(
    {
        "requeue",
        "_record_failure",
        "_record_pass",
        "_requeue_failed",
        "_schedule_retest",
    }
)


def _sets_status(function: ast.AST) -> bool:
    for item in ast.walk(function):
        if isinstance(item, ast.Assign) and any(
            isinstance(t, ast.Subscript) and _name(t) == "status" for t in item.targets
        ):
            return True
        if (
            isinstance(item, ast.Call)
            and _name(item.func) == "update"
            and any(k.arg == "status" for k in item.keywords)
        ):
            return True
    return False


def silent_lane_exits(source: str) -> list[str]:
    """Functions that move a recipe's result without writing a results row."""
    bad = []
    for function in _functions(ast.parse(source)):
        writes = bool(_calls(function, {"record_result"}))
        exit_without_row = function.name in LANE_EXITS and not writes
        silent_reset = _sets_status(function) and not (
            writes or _calls(function, LANE_EXITS)
        )
        if exit_without_row or silent_reset:
            bad.append(function.name)
    return bad


def stray_log_writers(source: str) -> list[str]:
    """``record_result`` is the one writer of results.jsonl."""
    return [
        function.name
        for function in _functions(ast.parse(source))
        if function.name != "record_result"
        and any(
            isinstance(call.func, ast.Attribute) and _name(call.func.value) == "log"
            for call in _calls(function, {"append"})
        )
    ]


def test_evidence_guard_finds_a_silent_requeue() -> None:
    silent = "def requeue(self, key, why):\n    self.entry(key)['status'] = 'pending'\n"
    assert silent_lane_exits(silent) == ["requeue"]
    reset = "def heal(self, entry):\n    entry.update(status='pending')\n"
    assert silent_lane_exits(reset) == ["heal"]
    via_exit = "def end(self, key):\n    self.requeue(key, 'x')\n"
    assert silent_lane_exits(via_exit) == []
    assert stray_log_writers("def f(self):\n    self.log.append(x=1)\n") == ["f"]


@pytest.mark.parametrize("path", SWEEP, ids=lambda p: p.name)
def test_every_lane_exit_writes_a_results_row(path: Path) -> None:
    source = path.read_text()
    assert silent_lane_exits(source) == [], str(path)
    assert stray_log_writers(source) == [], str(path)


def test_the_lane_exits_exist() -> None:
    source = (ROOT / "spark_sweep/run.py").read_text()
    names = {function.name for function in _functions(ast.parse(source))}
    assert LANE_EXITS | {"record_result"} <= names


# -- published vocabulary ----------------------------------------------------------------


def test_vocabulary_matches_published_controller_contract() -> None:
    published = json.loads((ROOT / "tests/fixtures/sweep-lifecycle.json").read_text())
    assert STATES == frozenset(published["LifecycleState"]["enum"])
    assert ACTIVE | TERMINAL | WAITING == STATES
    assert not ACTIVE & TERMINAL
    assert not WAITING & TERMINAL
    assert "backoff" in ACTIVE and "observing" in ACTIVE


def test_verdict_codes_match_the_published_enums() -> None:
    published = json.loads((ROOT / "tests/fixtures/sweep-vocabulary.json").read_text())
    words = {
        word
        for name, values in published.items()
        if name != "source"
        for word in values
    }
    from vonk_forge_contracts.sweep import SweepStorageCode

    used = {
        *SweepStorageCode,
        verdict.PHASE_RETRY,
        verdict.STALLED_CODE,
        verdict.NAS_SPACE_CODE,
        *verdict.PREPARATION_CODES,
        *verdict.CLIENT_WAIT_DECISIONS,
        verdict.CLIENT_EXIT_DECISION,
    }
    assert used <= words, sorted(used - words)


def dig_counts() -> dict[str, int]:
    counts = {}
    for path in sorted((ROOT / "spark_sweep").glob("*.py")):
        count = sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dig"
            for node in ast.walk(ast.parse(path.read_text()))
        )
        if count:
            counts[str(path.relative_to(ROOT))] = count
    return counts


def test_untyped_dig_access_only_falls() -> None:
    baseline = json.loads((ROOT / "tools/sweep-dig-baseline.json").read_text())
    counts = dig_counts()
    assert counts == baseline, (
        "dig access changed: remove new calls or tighten the baseline after a decrease"
    )


@pytest.mark.parametrize("word", sorted(STATES))
def test_fake_can_emit_every_lifecycle_word(tmp_path: Path, word: str) -> None:
    from sweep_fakes import FakeModel, FakeRecipe, make_sweep

    sweep, fleet, _ = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    fleet.observations.append((("recipe", "progress"), 0, {"id": "op", "state": word}))
    assert sweep.vk.call("recipe", "progress", "op")["state"] == word


# -- Testable is the enrolled topology, never the momentary online count ------------------

TOPOLOGY = frozenset(
    {
        "enrolled",
        "fleet_capacity",
        "_reconcile_all",
        "_reconcile_entry",
        "heal_fleet_skips",
    }
)


def online_reads(source: str, functions: frozenset[str] | None = None) -> list[str]:
    """Functions that decide from online Sparks: a ``sparks()`` call or an ``.online`` read."""
    return [
        function.name
        for function in _functions(ast.parse(source))
        if (functions is None or function.name in functions)
        and (
            _calls(function, {"sparks"})
            or any(
                isinstance(node, ast.Attribute) and node.attr == "online"
                for node in ast.walk(function)
            )
        )
    ]


def test_the_online_guard_finds_a_skip_from_the_online_count() -> None:
    bad = "def _reconcile_all(self):\n    usable = len(self.sparks())\n"
    assert online_reads(bad, TOPOLOGY) == ["_reconcile_all"]
    filtered = "def main(fleet):\n    return [s for s in fleet.sparks if s.online]\n"
    assert online_reads(filtered) == ["main"]


def test_testable_is_decided_from_enrolled_sparks_only() -> None:
    run = (ROOT / "spark_sweep/run.py").read_text()
    assert online_reads(run, TOPOLOGY) == []
    assert online_reads((ROOT / "spark_sweep/cli.py").read_text()) == []


def test_verdict_writers_do_not_select_identity_from_the_live_catalog() -> None:
    tree = ast.parse((ROOT / "spark_sweep/run.py").read_text())
    writers = {"_record_pass", "_record_failure", "_facts", "_download_done"}
    checked = set()
    for function in _functions(tree):
        if function.name not in writers:
            continue
        checked.add(function.name)
        assert not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "recipes"
            for node in ast.walk(function)
        ), f"{function.name} must take accepted identity from the lane/download record"
        assert _calls(function, {"read"}), (
            f"{function.name} must read the frozen record"
        )
    assert checked == writers


# -- S3: counters may pace retries, never settle uncertain work ----------------------


def count_to_terminal(source: str) -> list[tuple[str, int]]:
    counters = FATE_COUNTERS | {"rerequests", "max_attempts", "MAX_REREQUESTS"}

    def counted(node):
        return any(_name(part) in counters for part in ast.walk(node)) or any(
            isinstance(part, ast.Call)
            and _name(part.func) == "get"
            and part.args
            and isinstance(part.args[0], ast.Constant)
            and part.args[0].value in counters
            for part in ast.walk(node)
        )

    def terminal(node):
        return bool(_calls(node, ENDING | {"on_failure"})) or any(
            isinstance(part, ast.Constant)
            and part.value in ("deferred", "failed", "skipped")
            for part in ast.walk(node)
        )

    bad = []
    for function in _functions(ast.parse(source)):
        for item in ast.walk(function):
            if (
                isinstance(item, ast.Call)
                and _name(item.func) == "_record_failure"
                and any(keyword.arg == "final" for keyword in item.keywords)
            ):
                bad.append((function.name, item.lineno))
            if (
                isinstance(item, ast.If)
                and counted(item.test)
                and terminal(function)
                and (
                    any(isinstance(part, ast.Return) for part in ast.walk(item))
                    or terminal(item)
                )
            ):
                # A counted early return also gates later terminal fallthrough.
                bad.append((function.name, item.lineno))
            if (
                isinstance(item, ast.Return)
                and item.value is not None
                and counted(item.value)
                and any(isinstance(part, ast.Compare) for part in ast.walk(item.value))
            ):
                bad.append((function.name, item.lineno))
    return bad


def test_count_to_terminal_detector_catches_fallthrough_and_cache_abandonment():
    assert count_to_terminal(
        "def f(self, n):\n self._record_failure(failure, final=n > 0)\n"
    )
    assert count_to_terminal(
        "def f(e):\n if e['attempts'] < 2: return\n e.update(status='deferred')\n"
    )
    assert count_to_terminal(
        "def f(e):\n return e.get('rerequests', 0) < MAX_REREQUESTS\n"
    )
    assert not count_to_terminal(
        "def f(e):\n e['next_check'] = now + min(2 ** e['attempts'], 600)\n"
    )


@pytest.mark.parametrize("path", SWEEP, ids=lambda p: p.name)
def test_no_sweep_counter_can_terminate_uncertain_work(path):
    assert count_to_terminal(path.read_text()) == [], str(path)


def unowned_stream_reads(source: str) -> list[int]:
    """Blocking HTTP iteration must have semantic inactivity supervision."""
    bad = []
    for function in _functions(ast.parse(source)):
        for item in ast.walk(function):
            if (
                isinstance(item, (ast.For, ast.AsyncFor))
                and _name(item.iter) == "response"
            ):
                bad.append(item.lineno)
            if (
                isinstance(item, ast.Call)
                and _name(item.func) in {"readline", "read1"}
                and not _calls(function, {"last_progress"})
            ):
                bad.append(item.lineno)
    return bad


def test_stream_read_guard_rejects_socket_only_inactivity():
    assert unowned_stream_reads("def f(response):\n for line in response: pass\n")
    assert unowned_stream_reads("def f(response):\n return response.readline()\n")
    assert not unowned_stream_reads(
        "def f(response):\n if now - last_progress() > window: raise TimeoutError\n return response.read1(64)\n"
    )


@pytest.mark.parametrize("path", SWEEP, ids=lambda p: p.name)
def test_blocking_stream_reads_have_semantic_progress_evidence(path):
    assert unowned_stream_reads(path.read_text()) == [], str(path)


def test_wait_record_requires_a_contract_reason_and_finite_expiry_evidence():
    from pydantic import ValidationError
    from vonk_forge_contracts.sweep import SweepWait, SweepWaitExpiry, SweepWaitReason

    fields = {
        "wait": SweepWait(reason=SweepWaitReason.BLOCKER),
        "started_at": 0,
        "deadline": 600,
        "observed_at": 601,
    }
    assert SweepWaitExpiry.model_validate(fields).wait.reason is SweepWaitReason.BLOCKER
    for required in fields:
        with pytest.raises(ValidationError):
            SweepWaitExpiry.model_validate(
                {key: value for key, value in fields.items() if key != required}
            )


def waits_without_evidence(source: str) -> list[int]:
    bad = []
    for function in _functions(ast.parse(source)):
        writes = [
            item
            for item in ast.walk(function)
            if isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Subscript) and _name(target) == "waiting_on"
                for target in item.targets
            )
        ]
        writes += [
            item
            for item in ast.walk(function)
            if isinstance(item, ast.Call)
            and _name(item.func) == "update"
            and any(key.arg == "waiting_on" for key in item.keywords)
        ]
        if writes and function.name != "bounded_wait":
            bad.extend(item.lineno for item in writes)
    return bad


def test_wait_evidence_guard_rejects_free_text_only_holds():
    assert waits_without_evidence("def f(w):\n w['waiting_on'] = 'owner'\n")
    assert not waits_without_evidence(
        "def bounded_wait(w, wait):\n w.update(wait=wait.model_dump(), waiting_on=wait.description, wait_deadline=600)\n"
    )


@pytest.mark.parametrize("path", SWEEP, ids=lambda p: p.name)
def test_waits_must_carry_progress_evidence_and_a_typed_next_check(path):
    assert waits_without_evidence(path.read_text()) == [], str(path)
