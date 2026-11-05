"""Bounded git/gh evidence publication, isolated from hardware lane progress.

Each attempt gets a disposable clone. A competing push causes a fresh plan on
its next attempt; no force push, unbounded wait, or sweep-state write is used.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field
from vonk_forge_contracts.sweep import (
    SweepCheckConclusion,
    SweepCheckStatus,
    SweepCommitStatusState,
    SweepEvidenceExit,
    SweepEvidenceMergeState,
)

from .own_pulls import OWN_FIELDS, REPOSITORY, own_pulls
from .qualification import (
    backfill,
    catalog_recipe_keys,
    merge_evidence,
    prune_withdrawn_evidence,
    read_evidence,
    read_state_evidence,
    write_evidence,
)
from .state import write_atomic

BRANCH = "qualification/evidence"
TITLE = "qualification evidence"
_deadline: float | None = None
UNPUBLISHED_FILE = "unpublished.json"
PERSISTENT_BATCHES = 2


class EvidencePublicationTimeout(TimeoutError):
    """Only the optional repository transport ends; hardware lanes continue."""


_PASSED = (
    SweepCheckConclusion.SUCCESS,
    SweepCheckConclusion.NEUTRAL,
    SweepCheckConclusion.SKIPPED,
)
_STATUS_RED = (SweepCommitStatusState.FAILURE, SweepCommitStatusState.ERROR)


class CheckResult(BaseModel):
    """A check run (status, conclusion) or a commit status (state).

    Unknown provider words stay plain strings; only the shapes below are red.
    """

    model_config = ConfigDict(extra="ignore")
    status: SweepCheckStatus | str | None = None
    conclusion: SweepCheckConclusion | str | None = None
    state: SweepCommitStatusState | str | None = None

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

    @property
    def check_failed(self) -> bool:
        return any(check.failed for check in self.checks)


def command(cwd: Path, *args: str, timeout: int = 60) -> str:
    if _deadline is not None:
        remaining = _deadline - time.monotonic()
        if remaining <= 0:
            raise EvidencePublicationTimeout("evidence batch exceeded 240s")
        timeout = min(timeout, max(1, int(remaining)))
    result = subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=True
    )
    return result.stdout.strip()


def own_open_prs(cwd: Path) -> list[dict]:
    """Open PRs on the evidence branch that come from this repository."""
    own = own_pulls(
        json.loads(
            command(
                cwd,
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
                f"number,{OWN_FIELDS}",
            )
        )
    )
    if len(own) > 1:
        raise RuntimeError("multiple qualification evidence PRs; needs repair")
    return own


def sync(
    history: Path, directory: Path, state: Path | None = None
) -> SweepEvidenceExit:
    """One batch, one branch, one open PR; every subprocess has a deadline."""
    global _deadline
    _deadline = time.monotonic() + 240
    directory.mkdir(parents=True, exist_ok=True)
    (directory / UNPUBLISHED_FILE).unlink(
        missing_ok=True
    )  # only a synced batch reports
    opened = own_open_prs(directory)
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
                "number,mergeStateStatus,statusCheckRollup",
            )
        )
        # A PR that needs a base update is re-planned from fresh main. So is a
        # red PR that main has outgrown (the fix may be there); a red PR on
        # current main waits, visibly.
        red = details.check_failed
        replan = details.merge_state in (
            SweepEvidenceMergeState.BEHIND,
            SweepEvidenceMergeState.DIRTY,
        ) or (
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
                f"evidence PR #{pr.number} has a failing check; waiting for main to move"
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
        command(clone, "git", "add", "qualification/evidence")
        if not command(
            clone, "git", "diff", "origin/main", "--", "qualification/evidence"
        ):
            return (
                SweepEvidenceExit.SYNCED
            )  # merged evidence or an empty batch never creates an empty PR
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
            return SweepEvidenceExit.SYNCED
        opened = own_open_prs(clone)
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
            opened = own_open_prs(clone)
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
        return SweepEvidenceExit.SYNCED


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
        """Raised only when the same gap persists across batches: the publisher is broken."""
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
            self.message = (
                "evidence sync complete"
                if code == SweepEvidenceExit.SYNCED
                else "evidence PR awaiting CI/auto-merge; new results queued; check in 300s (see publisher.log)"
                if code == SweepEvidenceExit.WAITING
                else "evidence PR has a failing check; waiting for main to move (see publisher.log)"
                if code == SweepEvidenceExit.FAILING_CHECK
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
