"""The added lines of a unified diff, shared by the "reject new ..." tools.

The tools judge only lines a change adds. They take that change as diff text
(``--diff FILE|-``), so a test feeds them text and needs no repository. Git runs
once, at the edge that calls the tool: ``python3 tools/added-lines --base REV``
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
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

# The files the change-judging tools read.
ADDED_PATHSPEC = (
    "*.py",
    "tools/",
    "spark_sweep/",
    "src/",
    "scripts/",
    "tests/",
    "adapters/",
    "display/models/",
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
    # A rename is a deletion plus an addition, so it goes through the same
    # relocation and judged-file rules (an exempt owner renamed into scope
    # must not hide its content).
    "--no-renames",
)
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


def _kind(path: str) -> str:
    """The file type of a path: extensionless launchers and ``.py`` are both Python."""
    name = Path(path).name
    if "Dockerfile" in name:
        return "Dockerfile"
    return ".py" if Path(path).suffix in ("", ".py") else Path(path).suffix


def parse_added(diff: str, judged: Callable[[str], bool]) -> dict[str, set[int]]:
    """Map each file of a ``-U0`` diff to the line numbers it adds.

    A relocated block is not an addition: a deleted block may account for one
    identical added block (see ``_spend_relocated``), so a verbatim move or split
    does not re-surface old findings. ``judged`` is the calling guard's own
    predicate for the files it checks (scope and exempt owners): a deletion pays
    only when it judges both the file the block left and the file it entered, so a
    block leaving an exempt or out-of-scope file is still new work where it lands.
    Hunk counts decide what is content, so a line such as ``-- a/x`` is content.
    The evidence is the diff text alone, never history.
    """
    files: dict[str, set[int]] = {}
    added: dict[str, list[tuple[int, str]]] = {}
    removed: dict[str, list[str]] = {}
    path: str | None = None
    old_path = ""
    line = 0
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
    # Each deleted file is judged once, not once per target file.
    owners = [(owner, old) for owner, old in removed.items() if judged(owner)]
    for target, lines in added.items():
        if not judged(target):
            continue
        for owner, old in owners:
            if _kind(owner) == _kind(target):
                _spend_relocated(files[target], lines, old)
    return files


def _spend_relocated(
    target: set[int], lines: list[tuple[int, str]], old: list[str]
) -> None:
    """Drop from ``target`` each added block that ``old`` deleted unchanged.

    Ported from vonk-forge ``added_line_guards.added_lines``. Blank lines never
    align a match. A match splits where the indentation shift changes, spans at
    least two non-blank lines and three lines with the blanks that follow it, and
    needs one uniform shift. A deleted line is spent once. Then the rest of ``old``
    is matched in any order (reordered methods): there a block must be three
    consecutive lines with two non-blank (forge asks only three), so a lone common
    line plus blanks never hides new work.
    """
    old_lines = [(i, t) for i, t in enumerate(old) if t.strip()]
    new_lines = [(i, t) for i, (_, t) in enumerate(lines) if t.strip()]
    matches = SequenceMatcher(
        a=[t.lstrip() for _, t in old_lines],
        b=[t.lstrip() for _, t in new_lines],
        autojunk=False,
    ).get_matching_blocks()
    blocks: list[tuple[int, int, int]] = []
    for left, right, size in matches:
        start = 0
        previous = None
        for offset in range(size):
            shift = _indent(old_lines[left + offset][1]) - _indent(
                new_lines[right + offset][1]
            )
            if previous is not None and shift != previous:
                blocks.append((left + start, right + start, offset - start))
                start = offset
            previous = shift
        blocks.append((left + start, right + start, size - start))
    for left, right, size in blocks:
        if size < 2:
            continue
        old_idx = [i for i, _ in old_lines[left : left + size]]
        new_idx = [i for i, _ in new_lines[right : right + size]]
        if any(lines[i][0] not in target for i in range(new_idx[0], new_idx[-1] + 1)):
            continue
        old_end, new_end = old_idx[-1] + 1, new_idx[-1] + 1
        while old_end < len(old) and not old[old_end].strip():
            old_end += 1
        while new_end < len(lines) and not lines[new_end][1].strip():
            new_end += 1
        if min(old_end - old_idx[0], new_end - new_idx[0]) < 3:
            continue
        new_end = (
            new_idx[-1] + 1 + min(old_end - old_idx[-1] - 1, new_end - new_idx[-1] - 1)
        )
        for i in range(new_idx[0], new_end):
            target.discard(lines[i][0])
        for i in range(old_idx[0], old_end):
            old[i] = f"\0consumed:{i}"
    for _ in range(len(lines)):
        match = SequenceMatcher(
            a=old,
            b=[t if n in target else f"\0added:{n}" for n, t in lines],
            autojunk=False,
        ).find_longest_match()
        solid = sum(bool(old[match.a + i].strip()) for i in range(match.size))
        if match.size < 3 or solid < 2:
            break
        for i in range(match.size):
            target.discard(lines[match.b + i][0])
            old[match.a + i] = f"\0consumed:{match.a + i}"


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
