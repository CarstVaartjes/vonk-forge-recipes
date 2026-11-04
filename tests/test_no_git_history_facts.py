"""Git history is source control, never a store of facts (owner, 2026-11-04).

Measured facts live in qualification evidence, keyed by content identity. A tool
that reads an old revision's content (``git show <rev>:``, ``git log``,
``git archive``) must be a rule about the change or the source itself, listed here
with why; a new reader of history for facts fails this guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HISTORY = {"show", "log", "archive", "cat-file", "blame"}
GIT_HELPERS = {"git", "_git"}
ALLOWED = {
    # A change must not re-topologize a recipe: a rule on the change under review.
    "tools/check-recipe-topology": "base Spark count of the recipe a change edits",
    # The release index records when its source commit was made.
    "tools/build-catalog-index": "source commit time (release provenance)",
    # A merge tool for qualification definitions on a PR branch.
    "tools/migrate-pr-qualification": "three-way merge of definitions",
    # The one-time contracts 3.0.0 display split reads a revision archive.
    "tools/split-display": "one-time migration from a revision archive",
}


def history_reads(source: str) -> list[int]:
    """Lines where a git command that reads history is built."""
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.List | ast.Tuple):
            items = node.elts
            git = False
        elif isinstance(node, ast.Call):
            items = node.args
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            git = name in GIT_HELPERS
        else:
            continue
        for item in items:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                continue
            if item.value == "git":
                git = True
            elif git and item.value in HISTORY:
                found.append(node.lineno)
                break
    return found


def python_sources() -> list[Path]:
    paths: list[Path] = []
    for folder in ("qualification", "spark_sweep", "tools", "scripts"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            head = path.read_bytes()[:80]
            if path.suffix == ".py" or (
                not path.suffix
                and head.startswith(b"#!")
                and b"python" in head.split(b"\n")[0]
            ):
                paths.append(path)
    return paths


def test_no_tool_reads_facts_from_git_history() -> None:
    found = {
        str(path.relative_to(ROOT)): lines
        for path in python_sources()
        if (lines := history_reads(path.read_text()))
    }
    assert set(found) <= set(ALLOWED), {
        name: lines for name, lines in found.items() if name not in ALLOWED
    }


def test_every_allowance_is_still_needed() -> None:
    for name in ALLOWED:
        assert history_reads((ROOT / name).read_text()), f"{name}: drop its allowance"


@pytest.mark.parametrize(
    "source",
    [
        'subprocess.run(["git", "-C", root, "show", f"{base}:{path}"])',
        '_git(root, "show", f"{base}:{path}")',
        'git("log", "-p", "--", path)',
        'command(clone, "git", "log", "--format=%H")',
        '("git", "archive", revision)',
    ],
)
def test_the_guard_catches_every_form(source: str) -> None:
    assert history_reads(source)


@pytest.mark.parametrize(
    "source",
    [
        '_git(root, "diff", "--name-only", base)',
        'subprocess.run(["git", "ls-files", "-z"])',
        'git("rev-parse", "HEAD")',
        'print("show", "log")',
    ],
)
def test_source_control_plumbing_is_not_history(source: str) -> None:
    assert not history_reads(source)
