"""Tooling acts only on pull requests from this repository (a public repo takes fork PRs)."""

import ast
import json
from pathlib import Path
from unittest.mock import patch

from spark_sweep.own_pulls import OWN_FIELDS, own_pulls

ROOT = Path(__file__).resolve().parents[1]
OWN = {"isCrossRepository": False, "headRepositoryOwner": {"login": "CarstVaartjes"}}
FORK = {"isCrossRepository": True, "headRepositoryOwner": {"login": "someone-else"}}
HELPERS = {"own_pulls", "own_open_prs"}
ACTING = {"view", "merge", "close"}


def test_own_pulls_fails_closed():
    pulls = [
        {"number": 1, **OWN},
        {"number": 2, **FORK},
        {"number": 3, "isCrossRepository": False},  # owner missing
        {
            "number": 4,
            "headRepositoryOwner": {"login": "CarstVaartjes"},
        },  # flag missing
        {"number": 5, "isCrossRepository": False, "headRepositoryOwner": None},
        {
            "number": 6,
            "isCrossRepository": False,
            "headRepositoryOwner": {"login": "someone-else"},
        },
    ]
    assert [pr["number"] for pr in own_pulls(pulls)] == [1]


def _gh_verb(call: ast.Call) -> tuple[str, int] | None:
    strings = [
        a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None
        for a in call.args
    ]
    for i in range(len(strings) - 1):
        if strings[i] == "pr" and strings[i + 1] in {"list", *ACTING}:
            return strings[i + 1] or "", i
    return None


def unguarded_pr_calls(source: str) -> list[str]:
    """`gh pr list` must go through own_pulls with OWN_FIELDS; view/merge/close only
    inside a function that selects PRs through a helper."""
    tree = ast.parse(source)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    offenders = []

    def calls_helper(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in HELPERS
            for n in ast.walk(node)
        )

    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        found = _gh_verb(call)
        if found is None:
            continue
        verb, _ = found
        if verb == "list":
            node, wrapped = call, False
            while node in parents and not isinstance(node, ast.stmt):
                node = parents[node]
                wrapped |= (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "own_pulls"
                )
            fields = any(
                isinstance(n, ast.Name) and n.id == "OWN_FIELDS" for n in ast.walk(call)
            )
            if not (wrapped and fields):
                offenders.append(
                    f"line {call.lineno}: gh pr list not own_pulls+OWN_FIELDS"
                )
        else:
            node = call
            while node in parents and not isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef
            ):
                node = parents[node]
            if not calls_helper(node):
                offenders.append(f"line {call.lineno}: gh pr {verb} without a helper")
    return offenders


def _sources():
    for path in [*(ROOT / "tools").glob("*"), *(ROOT / "spark_sweep").glob("*.py")]:
        if not path.is_file():
            continue
        try:
            yield path, path.read_text()
        except UnicodeDecodeError:
            continue


def test_every_gh_pr_selection_goes_through_the_own_repo_helper():
    offenders = []
    for path, text in _sources():
        if "gh" not in text or '"pr"' not in text:
            continue
        try:
            found = unguarded_pr_calls(text)
        except SyntaxError:
            continue  # not python
        offenders += [f"{path.relative_to(ROOT)} {f}" for f in found]
    assert not offenders, offenders


def test_the_guard_fails_on_planted_unguarded_calls():
    listed = 'def f():\n    return gh_json("pr", "list", "--head", "x", "--json", "number")\n'
    wrapped_without_fields = (
        'def f():\n    return own_pulls(gh_json("pr", "list", "--json", "number"))\n'
    )
    merged = 'def f(n):\n    gh("pr", "merge", str(n), "--auto")\n'
    closed = 'def f(n):\n    command(cwd, "gh", "pr", "close", str(n))\n'
    for source in (listed, wrapped_without_fields, merged, closed):
        assert unguarded_pr_calls(source), source
    good = (
        "def f():\n"
        '    pulls = own_pulls(gh_json("pr", "list", "--json", f"number,{OWN_FIELDS}"))\n'
        '    gh("pr", "merge", str(pulls[0]["number"]))\n'
    )
    assert unguarded_pr_calls(good) == []


def _load_refresh():
    import importlib.machinery
    import importlib.util
    import sys

    loader = importlib.machinery.SourceFileLoader(
        "refresh_upstream", str(ROOT / "tools/refresh-upstream")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def test_refresh_upstream_never_arms_merges_or_closes_a_fork_pr():
    refresh = _load_refresh()
    pulls = [
        {
            "number": 7,
            "headRefName": "refresh/x",
            "autoMergeRequest": None,
            "body": "",
            "files": [],
            **FORK,
        },
        {
            "number": 8,
            "headRefName": "refresh/y",
            "autoMergeRequest": None,
            "body": "",
            "files": [],
            **OWN,
        },
    ]
    calls: list[tuple] = []

    def gh_json(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "pr":
            assert OWN_FIELDS in arguments[-1]
            return pulls
        return []

    def gh(*arguments: str, check: bool = True):
        calls.append(arguments)
        return json.dumps([])

    summary = refresh.Summary()
    summary.healthy = True
    with (
        patch("refresh_upstream.gh_json", side_effect=gh_json),
        patch("refresh_upstream.gh", side_effect=gh),
    ):
        refresh.arm_open_mechanical_prs(summary)
        refresh.close_stale_refresh_prs({}, summary)
        refresh.supersede_member_prs(["x", "z"], "z", 99, summary)
    acted = [c for c in calls if c[:2] in {("pr", "merge"), ("pr", "close")}]
    assert acted  # our own refresh/y was armed and closed
    assert all("7" not in c for c in acted)
    assert summary.armed == [8]
