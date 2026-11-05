"""Bounded git/gh evidence publication, isolated from hardware lane progress.

Each attempt gets a disposable clone. A competing push causes a fresh plan on
its next attempt; no force push, unbounded wait, or sweep-state write is used.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from vonk_forge_contracts import (
    QualificationEvidenceRecord,
    RecipeQualificationEvidence,
)
from vonk_forge_contracts.sweep import (
    PublisherState,
    SweepCheckConclusion,
    SweepCheckStatus,
    SweepCommitStatusState,
    SweepEvidenceExit,
    SweepEvidenceMergeState,
)

from .qualification import (
    backfill,
    catalog_recipe_keys,
    evidence_path,
    merge_evidence,
    prune_withdrawn_evidence,
    read_evidence,
    read_state_evidence,
    write_evidence,
)
from .state import write_atomic

BRANCH = "qualification/evidence"
TITLE = "qualification evidence"
REPOSITORY = "CarstVaartjes/vonk-forge-recipes"
_deadline: float | None = None
UNPUBLISHED_FILE = "unpublished.json"
PERSISTENT_BATCHES = 2
STATE_FILE = "publisher-state.json"


class EvidencePublicationTimeout(TimeoutError):
    """Only the optional repository transport ends; hardware lanes continue."""


_PASSED = (
    SweepCheckConclusion.SUCCESS,
    SweepCheckConclusion.NEUTRAL,
    SweepCheckConclusion.SKIPPED,
)
_STATUS_RED = (SweepCommitStatusState.FAILURE, SweepCommitStatusState.ERROR)
_STATUS_DONE = (*_STATUS_RED, SweepCommitStatusState.SUCCESS)


class PublisherRecord(BaseModel):
    """The persisted state file: what the last batch saw, and what is left out."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    state: PublisherState = PublisherState.IDLE
    signature: str = ""  # red check names, or the failing command and exit code
    head: str = ""  # the PR head the red was seen on: a repeat needs a new head
    command: bool = False  # the signature names a command rather than checks
    refused: tuple[str, ...] = ()  # record ids kept out of the publication
    refused_because: str = ""  # set when the refused set grows
    main: str = (
        ""  # origin/main when they were refused; they are released when it moves
    )


#: Declared edges. REFUSED is only reachable from a state that already saw red.
EDGES: dict[PublisherState, frozenset[PublisherState]] = {
    PublisherState.IDLE: frozenset({PublisherState.IDLE, PublisherState.RED}),
    PublisherState.RED: frozenset(PublisherState),
    PublisherState.REFUSED: frozenset(PublisherState),
}


def transition(
    current: PublisherRecord,
    *,
    signature: str | None,
    head: str = "",
    command: bool = False,
    refused: tuple[str, ...] | None = None,
    main: str | None = None,
    observed: bool = True,
) -> PublisherRecord:
    """The only place a record is chosen; an undeclared edge self-heals to IDLE.

    `signature` None means nothing failed. "Twice" means two different runs: the
    same red on a new PR head (after a re-plan push), or a command that failed on
    a new tick. The same head seen on two ticks is one run and stays RED. A
    rollup that is still running (`observed` False) changes nothing.
    """
    if not observed:
        signature, head, command = current.signature, current.head, current.command
        wanted = current.state
    elif signature is None:
        wanted = PublisherState.IDLE
    elif (
        current.state == PublisherState.RED
        and signature == current.signature
        and (command or head != current.head)
    ):
        wanted = PublisherState.REFUSED
    else:
        wanted = PublisherState.RED
    state = wanted if wanted in EDGES.get(current.state, ()) else PublisherState.IDLE
    kept = current.refused if refused is None else refused
    grew = bool(set(kept) - set(current.refused))
    red = state != PublisherState.IDLE
    return PublisherRecord(
        state=state,
        signature=signature or "" if red else "",
        head=head if red else "",
        command=command and red,
        refused=kept,
        refused_because=(
            (signature or "") if grew else current.refused_because if kept else ""
        ),
        main=(current.main if main is None else main) if kept else "",
    )


@dataclass
class Seen:
    """What one batch observed; sync() turns it into the persisted record."""

    signature: str | None = None
    head: str = ""
    command: bool = False
    refused: tuple[str, ...] | None = None
    main: str | None = None
    pending: bool = False  # the PR's checks are still running: nothing was observed
    closing: bool = False  # the blocked PR is being closed; a failure retries it


def load_state(directory: Path) -> PublisherRecord:
    """Missing, unreadable or unknown state is IDLE: the next batch re-observes."""
    try:
        return PublisherRecord.model_validate_json((directory / STATE_FILE).read_text())
    except (OSError, ValidationError):
        return PublisherRecord()


def record_id(key: str, record: QualificationEvidenceRecord) -> str:
    """A record's identity for the refused list; new content is a new id."""
    return f"{key}#{hashlib.sha256(record.model_dump_json().encode()).hexdigest()[:16]}"


class CheckResult(BaseModel):
    """A check run (status, conclusion) or a commit status (state).

    Unknown provider words stay plain strings; only the shapes below are red.
    """

    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    context: str | None = None
    status: SweepCheckStatus | str | None = None
    conclusion: SweepCheckConclusion | str | None = None
    state: SweepCommitStatusState | str | None = None

    @property
    def incomplete(self) -> bool:
        if self.status is not None:
            return self.status != SweepCheckStatus.COMPLETED
        return self.state is not None and self.state not in _STATUS_DONE

    @property
    def label(self) -> str:
        return self.name or self.context or "check"

    @property
    def failed(self) -> bool:
        if self.status is not None:
            return (
                self.status == SweepCheckStatus.COMPLETED
                and self.conclusion not in _PASSED
            )
        return self.state in _STATUS_RED


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    # Unknown provider states keep the current batch stable until it merges.
    merge_state: str = Field(
        default=SweepEvidenceMergeState.UNKNOWN, alias="mergeStateStatus"
    )
    checks: list[CheckResult] = Field(default_factory=list, alias="statusCheckRollup")
    head: str = Field(default="", alias="headRefOid")

    @property
    def check_failed(self) -> bool:
        return any(check.failed for check in self.checks)

    @property
    def settled(self) -> bool:
        """A fully completed rollup; a pending one is not an observation."""
        return not any(check.incomplete for check in self.checks)

    @property
    def failure_signature(self) -> str:
        """The red checks by name, never the commit they ran on."""
        return ",".join(sorted(check.label for check in self.checks if check.failed))


def command(cwd: Path, *args: str, timeout: int = 60) -> str:
    """One bounded subprocess. No retry here: the next tick re-plans from a fresh clone."""
    if _deadline is not None:
        remaining = _deadline - time.monotonic()
        if remaining <= 0:
            raise EvidencePublicationTimeout("evidence batch exceeded 240s")
        timeout = min(timeout, max(1, int(remaining)))
    result = subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=True
    )
    return result.stdout.strip()


def sync(
    history: Path, directory: Path, state: Path | None = None
) -> SweepEvidenceExit:
    """One batch, then record what it saw. A failing command never crashes the child."""
    seen = Seen()
    directory.mkdir(parents=True, exist_ok=True)
    previous = load_state(directory)
    try:
        code = _batch(history, directory, state, seen, previous)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        EvidencePublicationTimeout,
        ValueError,  # malformed evidence or provider JSON, incl. pydantic's ValidationError
    ) as error:
        # The signature is keyed on behaviour; the real error goes to publisher.log.
        print(f"evidence: {type(error).__name__}: {error}", flush=True)
        if stderr := getattr(error, "stderr", None):
            print(f"evidence: stderr: {stderr}", flush=True)
        if isinstance(error, ValueError):
            seen.signature = "validate evidence exit invalid"
        else:
            argv = " ".join(map(str, getattr(error, "cmd", ())[:2]))
            exit_code = getattr(error, "returncode", None)
            seen.signature = (
                f"{argv} exit {'timeout' if exit_code is None else exit_code}"
            )
        seen.command = True
        if seen.closing:  # the blocked PR is still open: observe it again next tick
            seen.signature, seen.head, seen.command = (
                previous.signature,
                previous.head,
                False,
            )
            seen.pending = False
            seen.refused = None
            print("evidence: closing the refused PR failed; retrying", flush=True)
        print(f"evidence: command failed: {seen.signature}", flush=True)
        code = SweepEvidenceExit.COMMAND_FAILED
    write_atomic(
        directory / STATE_FILE,
        transition(
            previous,
            signature=seen.signature,
            head=seen.head,
            command=seen.command,
            refused=seen.refused,
            main=seen.main,
            observed=not seen.pending,
        ).model_dump_json(),
    )
    return code


def _evidence_records(clone: Path, ref: str, name: str) -> tuple:
    try:
        return RecipeQualificationEvidence.model_validate_json(
            command(clone, "git", "show", f"{ref}:{name}")
        ).records
    except subprocess.CalledProcessError:
        return ()  # the file does not exist at that ref


def _branch_only_ids(clone: Path) -> list[str]:
    """Ids of the records the evidence branch carries that main does not."""
    command(clone, "git", "fetch", "origin", f"{BRANCH}:refs/remotes/origin/{BRANCH}")
    names = command(
        clone,
        "git",
        "diff",
        "--name-only",
        f"origin/main...origin/{BRANCH}",
        "--",
        "qualification/evidence",
    ).splitlines()
    ids = []
    for name in names:
        key = (
            Path(name).relative_to("qualification/evidence").with_suffix("").as_posix()
        )
        on_main = {
            r.model_dump_json() for r in _evidence_records(clone, "origin/main", name)
        }
        ids += [
            record_id(key, r)
            for r in _evidence_records(clone, f"origin/{BRANCH}", name)
            if r.model_dump_json() not in on_main
        ]
    return ids


def _drop_refused(clone: Path, refused: set[str]) -> tuple[str, ...]:
    """Remove refused records from the clone; return the ones still produced locally.

    Only the files named by the refused ids are read.
    """
    found: list[str] = []
    for key in sorted({i.split("#")[0] for i in refused}):
        evidence = read_evidence(clone, key)
        ids = [record_id(key, r) for r in evidence.records]
        dropped = [i for i in ids if i in refused]
        if not dropped:
            continue
        found += dropped
        kept = tuple(r for r, i in zip(evidence.records, ids) if i not in refused)
        if kept:
            write_evidence(clone, evidence.model_copy(update={"records": kept}))
        else:
            evidence_path(clone, key).unlink()
    return tuple(sorted(found))


def _batch(
    history: Path,
    directory: Path,
    state: Path | None,
    seen: Seen,
    previous: PublisherRecord,
) -> SweepEvidenceExit:
    """One batch, one branch, one open PR; every subprocess has a deadline."""
    global _deadline
    _deadline = time.monotonic() + 240
    directory.mkdir(parents=True, exist_ok=True)
    (directory / UNPUBLISHED_FILE).unlink(
        missing_ok=True
    )  # only a synced batch reports
    opened = json.loads(
        command(
            directory,
            "gh",
            "pr",
            "list",
            "--repo",
            REPOSITORY,
            "--head",
            BRANCH,
            "--state",
            "open",
            "--json",
            "number",
        )
    )
    if len(opened) > 1:
        raise RuntimeError("multiple qualification evidence PRs; needs repair")
    refusing = False
    blocked = 0
    if opened:
        pr = PullRequest.model_validate(opened[0])
        details = PullRequest.model_validate_json(
            command(
                directory,
                "gh",
                "pr",
                "view",
                str(pr.number),
                "--repo",
                REPOSITORY,
                "--json",
                "number,mergeStateStatus,statusCheckRollup,headRefOid",
            )
        )
        # A PR that needs a base update is re-planned from fresh main. So is a
        # red PR that main has outgrown (the fix may be there) and one refused
        # (the same failure twice): new evidence still flows. A red PR seen
        # once on current main waits, visibly.
        red = details.check_failed
        seen.signature = details.failure_signature if red else None
        seen.head = details.head
        # Checks still running, or none registered yet on a head we have not seen red.
        seen.pending = not red and (
            not details.settled
            or (not details.checks and details.head != previous.head)
        )
        blocked = pr.number
        refusing = (
            transition(previous, signature=seen.signature, head=seen.head).state
            == PublisherState.REFUSED
        )
        replan = (
            details.merge_state
            in (
                SweepEvidenceMergeState.BEHIND,
                SweepEvidenceMergeState.DIRTY,
            )
            or refusing
            or (
                red
                and int(
                    command(
                        directory,
                        "gh",
                        "api",
                        f"repos/{REPOSITORY}/compare/main...{BRANCH}",
                        "--jq",
                        ".behind_by",
                    )
                )
            )
        )
        if not replan:
            command(
                directory,
                "gh",
                "pr",
                "merge",
                str(pr.number),
                "--repo",
                REPOSITORY,
                "--auto",
                "--squash",
            )
            print(
                f"evidence PR #{pr.number} has a failing check ({seen.signature}); waiting for main to move"
                if red
                else f"evidence: PR #{pr.number} awaiting CI/auto-merge; new results remain in history",
                flush=True,
            )
            return SweepEvidenceExit.FAILING_CHECK if red else SweepEvidenceExit.WAITING
        # Conflicts or required base updates get a fresh clone and a new plan.
    with tempfile.TemporaryDirectory(
        prefix="qualification-", dir=directory
    ) as temporary:
        clone = Path(temporary)
        done = SweepEvidenceExit.REFUSED if refusing else SweepEvidenceExit.SYNCED
        command(
            directory,
            "git",
            "clone",
            "--quiet",
            "--no-checkout",
            "--single-branch",
            "--branch",
            "main",
            f"https://github.com/{REPOSITORY}.git",
            str(clone),
            timeout=120,
        )
        remote = command(clone, "git", "ls-remote", "--heads", "origin", BRANCH)
        seen.main = command(clone, "git", "rev-parse", "origin/main")
        # Refused records are released when main moves: the red may be fixed there.
        carried = previous.refused if previous.main == seen.main else ()
        seen.refused = carried
        if refusing:
            # The same red checks on two runs: close the blocked PR and plan from
            # main without its records, so new evidence is no longer held behind it.
            refused = tuple(sorted({*carried, *_branch_only_ids(clone)}))
            seen.closing = True
            command(
                clone,
                "gh",
                "pr",
                "close",
                str(blocked),
                "--repo",
                REPOSITORY,
                "--delete-branch",
                "--comment",
                f"Refused: the same checks stayed red twice ({seen.signature}). "
                "Its records are left out of the next evidence PR until their content changes.",
            )
            seen.closing = False
            seen.refused = refused  # recorded only once the close succeeded
            remote = ""
        if remote:
            command(
                clone,
                "git",
                "fetch",
                "origin",
                f"{BRANCH}:refs/remotes/origin/{BRANCH}",
            )
            command(clone, "git", "checkout", "-b", BRANCH, f"origin/{BRANCH}")
            # Read both parent versions before merge; evidence conflicts are unions.
            inherited = {}
            for path in (clone / "qualification/evidence").rglob("*.json"):
                key = (
                    path.relative_to(clone / "qualification/evidence")
                    .with_suffix("")
                    .as_posix()
                )
                inherited[key] = read_evidence(clone, key)
            result = subprocess.run(
                ["git", "merge", "--no-commit", "--no-ff", "origin/main"],
                cwd=clone,
                capture_output=True,
                text=True,
                timeout=min(60, max(1, int(_deadline - time.monotonic()))),
                check=False,
            )
            if result.returncode:
                conflicts = command(
                    clone, "git", "diff", "--name-only", "--diff-filter=U"
                ).splitlines()
                if not conflicts or any(
                    not p.startswith("qualification/evidence/") for p in conflicts
                ):
                    raise RuntimeError(
                        "evidence branch has a non-evidence conflict; needs repair"
                    )
                keys = catalog_recipe_keys(clone)
                for name in conflicts:
                    key = (
                        Path(name)
                        .relative_to("qualification/evidence")
                        .with_suffix("")
                        .as_posix()
                    )
                    if key not in keys:
                        command(clone, "git", "rm", "--", name)
                    else:
                        command(clone, "git", "checkout", "--theirs", "--", name)
            keys = catalog_recipe_keys(clone)
            for key, evidence in inherited.items():
                if key not in keys:
                    continue
                write_evidence(
                    clone, merge_evidence(read_evidence(clone, key), evidence)
                )
        else:
            command(clone, "git", "checkout", "-b", BRANCH, "origin/main")
        paths = sorted(history.parent.glob(history.name + ".*"))
        if history.exists():
            paths.append(history)
        local = read_state_evidence(state) if state is not None else {}
        result = backfill(paths, clone, local)
        # Counted before the merge, on the fresh clone; reported, never a blocker.
        write_atomic(
            directory / UNPUBLISHED_FILE, json.dumps({"count": result.unpublished})
        )
        if result.rejected:
            print(
                f"evidence: rejected {result.rejected} malformed historical rows",
                flush=True,
            )
        prune_withdrawn_evidence(clone)
        seen.refused = _drop_refused(clone, set(seen.refused or ()))
        command(clone, "git", "add", "qualification/evidence")
        if not command(
            clone, "git", "diff", "origin/main", "--", "qualification/evidence"
        ):
            return done  # merged evidence or an empty batch never creates an empty PR
        changed = command(clone, "git", "diff", "--cached", "--name-only")
        merging = (clone / ".git/MERGE_HEAD").exists()
        if changed or merging:
            command(
                clone,
                "git",
                "commit",
                "--no-verify",
                "-m",
                "Record hardware qualification evidence",
            )
            command(clone, "git", "push", "origin", f"HEAD:refs/heads/{BRANCH}")
        elif not remote:
            return done
        opened = json.loads(
            command(
                clone,
                "gh",
                "pr",
                "list",
                "--repo",
                REPOSITORY,
                "--head",
                BRANCH,
                "--state",
                "open",
                "--json",
                "number",
            )
        )
        if len(opened) > 1:
            raise RuntimeError("multiple qualification evidence PRs; needs repair")
        if not opened:
            command(
                clone,
                "gh",
                "pr",
                "create",
                "--repo",
                REPOSITORY,
                "--head",
                BRANCH,
                "--base",
                "main",
                "--title",
                TITLE,
                "--body",
                "Sanitized hardware evidence from the maintainer sweep. Status is derived at library publication.",
            )
            opened = json.loads(
                command(
                    clone,
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    REPOSITORY,
                    "--head",
                    BRANCH,
                    "--state",
                    "open",
                    "--json",
                    "number",
                )
            )
        pr = PullRequest.model_validate(opened[0])
        command(
            clone,
            "gh",
            "pr",
            "merge",
            str(pr.number),
            "--repo",
            REPOSITORY,
            "--auto",
            "--squash",
        )
        print(f"evidence: PR #{pr.number}, auto-merge requested", flush=True)
        return done


class EvidencePublisher:
    """Tick-driven child process; never wait on git or GitHub in a sweep lane."""

    def __init__(
        self,
        history: Path,
        directory: Path,
        script: Path,
        state: Path | None = None,
    ) -> None:
        self.history = history
        self.state = state
        self.unpublished = 0
        self.unpublished_batches = 0
        self.directory = directory
        self.script = script
        self.process: subprocess.Popen | None = None
        self.next_attempt = 0.0
        self.started_at = 0.0
        self.message = "evidence publication pending"
        self.record = PublisherRecord()
        self._output: TextIO | None = None

    def _read_unpublished(self) -> None:
        try:
            count = int(
                json.loads((self.directory / UNPUBLISHED_FILE).read_text())["count"]
            )
        except (OSError, ValueError, KeyError, TypeError):
            return  # this batch did not reach backfill: neither counted nor reset
        self.unpublished = count
        self.unpublished_batches = self.unpublished_batches + 1 if count else 0
        if count:
            self.message += f"; {count} resolved sweep records were missing from main"

    @property
    def alert(self) -> str | None:
        """Raised when the publisher is broken: a refused failure, or a gap that persists."""
        if self.record.refused:
            return (
                f"evidence publisher: {len(self.record.refused)} records refused and left "
                f"out of the publication (red twice: {self.record.refused_because}); "
                "they return when their content changes"
            )
        if self.record.state == PublisherState.REFUSED:
            return f"evidence publisher: failing repeatedly: {self.record.signature}"
        if self.unpublished_batches < PERSISTENT_BATCHES:
            return None
        return (
            f"evidence publisher: {self.unpublished} resolved sweep records still "
            f"missing from main after {self.unpublished_batches} batches"
        )

    def tick(self, now: float) -> None:
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                if now - self.started_at > 240:
                    self.process.kill()  # only our child PID; the next tick reaps it
                    self.message = "evidence publication exceeded 240s; retry scheduled"
                return
            self.process.wait(timeout=1)
            self.record = load_state(self.directory)
            signature = self.record.signature
            self.message = (
                "evidence sync complete"
                if code == SweepEvidenceExit.SYNCED
                else "evidence PR awaiting CI/auto-merge; new results queued; check in 300s (see publisher.log)"
                if code == SweepEvidenceExit.WAITING
                else f"evidence PR has failing checks ({signature}); waiting for main to move (see publisher.log)"
                if code == SweepEvidenceExit.FAILING_CHECK
                else (
                    f"evidence publisher command failed ({signature}); re-plan in 300s (see publisher.log)"
                    if self.record.command
                    else f"evidence publisher: closing the refused PR failed, retrying (last red: {signature}; see publisher.log)"
                )
                if code == SweepEvidenceExit.COMMAND_FAILED
                else f"evidence PR refused (red twice: {signature}); {len(self.record.refused)} records left out, a fresh PR carries the rest (see publisher.log)"
                if code == SweepEvidenceExit.REFUSED
                else f"evidence publisher exited {code}; re-plan in 300s (see publisher.log)"
            )
            self._read_unpublished()
            if self._output is not None:
                self._output.close()
            self.process = None
            self.next_attempt = now + 300
        if now < self.next_attempt:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._output = (self.directory / "publisher.log").open("a")
            import sys

            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.script),
                    "--history",
                    str(self.history),
                    "--work-dir",
                    str(self.directory),
                    *(["--state", str(self.state)] if self.state else []),
                ],
                stdout=self._output,
                stderr=self._output,
            )
        except OSError as error:
            if self._output is not None:
                self._output.close()
            self.process = None
            self.next_attempt = now + 300
            self.message = f"evidence publisher unavailable ({type(error).__name__}); retry in 300s"
            return
        self.started_at = now
        self.message = "evidence publication running (240s budget)"
