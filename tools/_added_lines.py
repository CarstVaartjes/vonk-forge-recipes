"""The added lines of a unified diff, shared by the "reject new ..." tools.

The tools judge only lines a change adds. They take that change as diff text
(``--diff FILE|-``), so a test feeds them text and needs no repository. Git runs
once, at the edge that calls the tool: the ``validate.yml`` step writes the diff
and the pre-commit hook pipes it. ``git_diff`` is the one helper for a developer
running a tool by hand with ``--base``; tests never call it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# The files the change-judging tools read. validate.yml writes the shared diff
# with exactly this pathspec (a test keeps the two equal).
ADDED_PATHSPEC = (
    "*.py",
    "tools/",
    "spark_sweep/",
    "qualification/",
    "scripts/",
    "tests/",
)
DIFF_ARGUMENTS = ("diff", "-U0", "--no-color", "--diff-filter=AMR")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_added(diff: str) -> dict[str, set[int]]:
    """Map each file of a ``-U0`` diff to the line numbers it adds."""
    result: dict[str, set[int]] = {}
    path: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].removeprefix("b/") if line != "+++ /dev/null" else None
        elif path and (hunk := HUNK.match(line)):
            start, count = int(hunk[1]), int(hunk[2] if hunk[2] is not None else 1)
            result.setdefault(path, set()).update(range(start, start + count))
    return result


def read_diff(source: str) -> str:
    """The diff text in ``source``: a file path, or ``-`` for standard input."""
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")


def git_diff(root: Path, base: str) -> str:
    """The diff a developer's ``--base REV`` stands for (the edge, never a test)."""
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            *DIFF_ARGUMENTS,
            f"{base}...HEAD",
            "--",
            *ADDED_PATHSPEC,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.exit(f"git diff {base}...HEAD: {completed.stderr.strip()}")
    return completed.stdout
