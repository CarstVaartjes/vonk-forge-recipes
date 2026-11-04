"""Git history is source control, never a store of facts (owner, 2026-11-04).

Measured facts live in qualification evidence, keyed by content identity. Code
that reads an old revision's content (``git show <rev>:``, ``git log``,
``git archive`` and the like) must be a rule about the change or the source
itself, allowed here per function with why; a new reader of history fails.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ("show", "log", "archive", "cat-file", "blame", "rev-list", "restore")
# A git command naming a history verb on one source line, in any form: an argv
# list, a starred prefix, a shell string or a helper's arguments.
VERBS = "|".join(map(re.escape, HISTORY))
HISTORY_LINE = re.compile(rf"""\bgit\b[^#\n]*?["'\s(,](?:{VERBS})["'\s,)]""")
ALLOWED = {
    # A change must not re-topologize a recipe: a rule on the change under review.
    (
        "tools/check-recipe-topology",
        "base_recipes",
    ): "base Spark count of edited recipes",
    # The release index records when its source commit was made.
    ("tools/build-catalog-index", "_commit_time"): "source commit time (provenance)",
    # A merge tool for qualification definitions on a PR branch.
    ("tools/migrate-pr-qualification", "main"): "three-way merge of definitions",
    # The one-time contracts 3.0.0 display split reads a revision archive.
    ("tools/split-display", "main"): "one-time migration from a revision archive",
}


def _scopes(tree: ast.AST) -> list[tuple[int, int, str]]:
    return [
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _scope(scopes: list[tuple[int, int, str]], line: int) -> str:
    inner = [s for s in scopes if s[0] <= line <= s[1]]
    return min(inner, key=lambda s: s[1] - s[0])[2] if inner else "<module>"


def _ast_lines(tree: ast.AST) -> set[int]:
    """Calls and argv sequences that pass ``git`` then a history verb."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.List | ast.Tuple):
            items, git = node.elts, False
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            items, git = node.args, "git" in name.lower()
        elif isinstance(node, ast.BinOp) and isinstance(node.right, ast.List):
            # ``GIT + ["show", ref]``: a git argv prefix joined to the verb.
            items = node.right.elts
            git = "git" in ast.unparse(node.left).lower()
        else:
            continue
        for item in items:
            if isinstance(item, ast.Starred) and "git" in ast.unparse(item).lower():
                git = True
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                continue
            if item.value == "git" or item.value.startswith("git "):
                git = True
                if any(f" {verb}" in f" {item.value} " for verb in HISTORY):
                    found.add(node.lineno)
            elif git and item.value in HISTORY:
                found.add(node.lineno)
                break
    return found


def history_reads(source: str) -> list[tuple[int, str]]:
    """``(line, function)`` of every git command that reads history."""
    tree = ast.parse(source)
    lines = _ast_lines(tree) | {
        number
        for number, text in enumerate(source.splitlines(), start=1)
        if HISTORY_LINE.search(text) and not text.lstrip().startswith("#")
    }
    scopes = _scopes(tree)
    return sorted((line, _scope(scopes, line)) for line in lines)


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
        (str(path.relative_to(ROOT)), scope): line
        for path in python_sources()
        for line, scope in history_reads(path.read_text())
    }
    assert set(found) <= set(ALLOWED), {
        site: line for site, line in found.items() if site not in ALLOWED
    }


def test_every_allowance_is_still_needed() -> None:
    for name, scope in ALLOWED:
        scopes = {s for _, s in history_reads((ROOT / name).read_text())}
        assert scope in scopes, f"{name}:{scope}: drop its allowance"


@pytest.mark.parametrize(
    "source",
    [
        'subprocess.run(["git", "-C", root, "show", f"{base}:{path}"])',
        '_git(root, "show", f"{base}:{path}")',
        'git("log", "-p", "--", path)',
        'run_git("rev-list", base)',
        'git_output(root, "cat-file", "-p", ref)',
        'command(clone, "git", "log", "--format=%H")',
        '("git", "archive", revision)',
        'subprocess.run([*GIT, "show", ref])',
        'subprocess.run(GIT + ["show", ref])',
        'gitcmd("show", ref)',
        'subprocess.run(f"git show {base}:{path}", shell=True)',
        'shlex.split("git log -p --follow x")',
        'subprocess.run(["git", "restore", "--source", base, path])',
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
        "# git show is never used here",
        'command(clone, "git", "checkout", "--theirs", "--", name)',
    ],
)
def test_source_control_plumbing_is_not_history(source: str) -> None:
    assert not history_reads(source)


def test_an_allowance_covers_one_function_only() -> None:
    source = (
        "def base_recipes(root):\n"
        '    return _git(root, "show", "a:b")\n'
        "def peaks(root):\n"
        '    return _git(root, "show", "a:c")\n'
    )
    assert [scope for _, scope in history_reads(source)] == ["base_recipes", "peaks"]
