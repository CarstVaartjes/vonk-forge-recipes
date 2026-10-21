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
from vonk_forge_contracts.sweep import SweepEvidenceExit, SweepEvidenceMergeState

from .qualification import backfill, merge_evidence, read_evidence, write_evidence

BRANCH = "qualification/evidence"
TITLE = "qualification evidence"
REPOSITORY = "CarstVaartjes/vonk-forge-recipes"
_deadline: float | None = None


class EvidencePublicationTimeout(TimeoutError):
    """Only the optional repository transport ends; hardware lanes continue."""


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    # Unknown provider states keep the current batch stable until it merges.
    merge_state: str = Field(
        default=SweepEvidenceMergeState.UNKNOWN, alias="mergeStateStatus"
    )


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


def sync(history: Path, directory: Path) -> bool:
    """One batch, one branch, one open PR; every subprocess has a deadline."""
    global _deadline
    _deadline = time.monotonic() + 240
    directory.mkdir(parents=True, exist_ok=True)
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
                "number,mergeStateStatus",
            )
        )
        if details.merge_state not in (
            SweepEvidenceMergeState.BEHIND,
            SweepEvidenceMergeState.DIRTY,
        ):
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
                f"evidence: PR #{pr.number} awaiting CI/auto-merge; new results remain in history",
                flush=True,
            )
            return False
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
                for name in conflicts:
                    command(clone, "git", "checkout", "--theirs", "--", name)
            for key, evidence in inherited.items():
                write_evidence(
                    clone, merge_evidence(read_evidence(clone, key), evidence)
                )
        else:
            command(clone, "git", "checkout", "-b", BRANCH, "origin/main")
        paths = sorted(history.parent.glob(history.name + ".*"))
        if history.exists():
            paths.append(history)
        result = backfill(paths, clone)
        if result.rejected:
            print(
                f"evidence: rejected {result.rejected} malformed historical rows",
                flush=True,
            )
        command(clone, "git", "add", "qualification/evidence")
        if not command(
            clone, "git", "diff", "origin/main", "--", "qualification/evidence"
        ):
            return True  # merged evidence or an empty batch never creates an empty PR
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
            return True
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
        return True


class EvidencePublisher:
    """Tick-driven child process; never wait on git or GitHub in a sweep lane."""

    def __init__(self, history: Path, directory: Path, script: Path) -> None:
        self.history = history
        self.directory = directory
        self.script = script
        self.process: subprocess.Popen | None = None
        self.next_attempt = 0.0
        self.started_at = 0.0
        self.message = "evidence publication pending"
        self._output: TextIO | None = None

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
                else f"evidence publisher exited {code}; re-plan in 300s (see publisher.log)"
            )
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
