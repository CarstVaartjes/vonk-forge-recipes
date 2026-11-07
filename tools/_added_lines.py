"""The added lines of a unified diff, shared by the "reject new ..." tools.

The tools judge only lines a change adds. They take that change as diff text
(``--diff FILE|-``), so a test feeds them text and needs no repository. Git runs
once, at the edge that calls the tool: ``python3 tools/_added_lines.py --base REV``
(or ``--cached`` for the staged change) writes the diff, and the validate.yml step
and the pre-commit hook call it. This module is the one owner of the diff flags and
the pathspec. ``git_diff`` also backs the developer shortcut ``--base`` of the
tools that still take it; tests never call it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The files the change-judging tools read.
ADDED_PATHSPEC = (
    "*.py",
    "tools/",
    "spark_sweep/",
    "qualification/",
    "scripts/",
    "tests/",
)
# Explicit prefixes: a developer's diff.mnemonicPrefix or diff.noprefix must not
# change how paths are parsed.
DIFF_ARGUMENTS = (
    "diff",
    "-U0",
    "--no-color",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "--diff-filter=AMR",
)
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


def git_diff(root: Path, base: str | None = None) -> str:
    """The added-lines diff of ``base...HEAD``, or of the staged change if no base.

    The edge, never a test: it is the only place git runs for these tools.
    """
    range_arguments = [f"{base}...HEAD"] if base else ["--cached"]
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            *DIFF_ARGUMENTS,
            *range_arguments,
            "--",
            *ADDED_PATHSPEC,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.exit(f"git diff {' '.join(range_arguments)}: {completed.stderr.strip()}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the added-lines diff.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="diff HEAD against the merge base with REV")
    source.add_argument("--cached", action="store_true", help="the staged change")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    sys.stdout.write(git_diff(arguments.root, arguments.base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
