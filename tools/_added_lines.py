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
from difflib import SequenceMatcher
from pathlib import Path

# The files the change-judging tools read.
ADDED_PATHSPEC = (
    "*.py",
    "tools/",
    "spark_sweep/",
    "qualification/",
    "scripts/",
    "tests/",
    "adapters/*Dockerfile*",
)
# Explicit prefixes: a developer's diff.mnemonicPrefix or diff.noprefix must not
# change how paths are parsed.
DIFF_ARGUMENTS = (
    "diff",
    "-U0",
    "--no-color",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "--diff-filter=AMRD",
)
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


def parse_added(diff: str) -> dict[str, set[int]]:
    """Map each file of a ``-U0`` diff to the line numbers it adds.

    A relocated block is not an addition: a deleted block may account for one
    identical added block in a file of the same type, so a verbatim move or split
    does not re-surface old findings. Copies and edited lines stay additions, the
    match needs a block (not an isolated common line), and it tolerates one uniform
    indentation shift. The evidence is the diff text alone, never history.
    """
    files: dict[str, set[int]] = {}
    added: dict[str, list[tuple[int, str]]] = {}
    removed: dict[str, list[str]] = {}
    path: str | None = None
    old_path = ""
    line = 0
    # Lines the open hunk still owes: a content line that reads like a file header
    # ("--- a/x") is content while the hunk counts say so.
    owed_old = owed_new = 0
    for text in diff.splitlines():
        in_hunk = owed_old > 0 or owed_new > 0
        if text.startswith("diff --git "):
            owed_old = owed_new = 0
        elif in_hunk and text.startswith("+"):
            if path:
                added[path].append((line, text[1:]))
            line += 1
            owed_new -= 1
        elif in_hunk and text.startswith("-"):
            removed.setdefault(old_path, []).append(text[1:])
            owed_old -= 1
        elif text.startswith("--- ") and not in_hunk:
            old_path = text[4:].removeprefix("a/")
        elif text.startswith("+++ ") and not in_hunk:
            path = text[4:].removeprefix("b/") if text != "+++ /dev/null" else None
            if path:
                added.setdefault(path, [])
        elif hunk := HUNK.match(text):
            start, count = int(hunk[3]), int(hunk[4] if hunk[4] is not None else 1)
            owed_old = int(hunk[2] if hunk[2] is not None else 1)
            owed_new, line = count, start
            if path:
                files.setdefault(path, set()).update(range(start, start + count))
    for target, lines in added.items():
        for owner, old in removed.items():
            if Path(owner).suffix == Path(target).suffix:
                _spend_relocated(files[target], lines, old)
    return files


def _spend_relocated(
    target: set[int], lines: list[tuple[int, str]], old: list[str]
) -> None:
    """Drop from ``target`` each added block that ``old`` deleted unchanged."""
    # A block is three lines with two non-blank, never an isolated common line.
    consumed = "\0consumed"
    pairs = SequenceMatcher(
        a=[text.lstrip() for text in old],
        b=[text.lstrip() for _, text in lines],
        autojunk=False,
    ).get_matching_blocks()
    for left, right, size in pairs:
        shifts = {
            _indent(old[left + i]) - _indent(lines[right + i][1])
            for i in range(size)
            if old[left + i].strip()
        }
        solid = sum(bool(old[left + i].strip()) for i in range(size))
        if size < 3 or solid < 2 or len(shifts) != 1:
            continue
        if not all(lines[right + i][0] in target for i in range(size)):
            continue
        for i in range(size):
            target.discard(lines[right + i][0])
            old[left + i] = consumed
    # Reordered extracted methods need not keep the old sequence alignment:
    # spend the remaining exact blocks independently.
    for _ in range(len(lines)):
        match = SequenceMatcher(
            a=old,
            b=[
                text if number in target else f"\0added:{number}"
                for number, text in lines
            ],
            autojunk=False,
        ).find_longest_match()
        solid = sum(bool(old[match.a + i].strip()) for i in range(match.size))
        if match.size < 3 or solid < 2:
            break
        for i in range(match.size):
            target.discard(lines[match.b + i][0])
            old[match.a + i] = consumed + str(match.a + i)


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
