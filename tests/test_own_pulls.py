"""Tooling acts only on pull requests from this repository (a public repo takes fork PRs)."""

import ast
import json
import re
from pathlib import Path
from unittest.mock import patch

from spark_sweep.own_pulls import OWN_FIELDS, own_pulls

ROOT = Path(__file__).resolve().parents[1]
OWN = {"isCrossRepository": False, "headRepositoryOwner": {"login": "CarstVaartjes"}}
FORK = {"isCrossRepository": True, "headRepositoryOwner": {"login": "someone-else"}}
WRAPPERS = {"own_pulls", "with_all_files"}
HELPERS = {*WRAPPERS, "own_open_prs"}
ACTING = {
    "view",
    "merge",
    "close",
    "edit",
    "comment",
    "reopen",
    "ready",
    "review",
    "checks",
    "diff",
}
MUTATING = {"-X", "--method", "-f", "-F", "--field", "--raw-field", "--input"}
BY_NUMBER = re.compile(r"^\d+$")


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


def _tokens(items: list[ast.expr]) -> list[str | None]:
    """String constants of the given expressions in order, looking inside lists, tuples
    and f-strings; anything computed is None."""
    out: list[str | None] = []
    stack = list(items)
    while stack:
        item = stack.pop(0)
        if isinstance(item, ast.List | ast.Tuple):
            stack = [*item.elts, *stack]
        elif isinstance(item, ast.Starred):
            stack = [item.value, *stack]
        elif isinstance(item, ast.Constant) and isinstance(item.value, str):
            out.append(item.value)
        elif isinstance(item, ast.JoinedStr):
            out.append(
                "".join(
                    v.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    else "\0"
                    for v in item.values
                )
            )
        else:
            out.append(None)
    return out


def _is_helper_call(n: ast.AST, names=HELPERS) -> bool:
    return (
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
    )


def unguarded_pr_calls(source: str) -> list[str]:
    """What the own-repo rule forbids. Every call and every list/tuple literal in the
    file is read as a possible `gh` argv (positional and keyword arguments, and argv
    lists built into a variable first):

    * `gh pr list` unless its result reaches `own_pulls` (wrapped, or assigned to a
      variable whose every use is an argument of `own_pulls`) and asks for OWN_FIELDS;
    * `gh pr view|merge|close|edit|review|checks|diff|...`, `gh api` on `/pulls` or
      graphql, and a mutating `gh api` on `issues/` (the issues API can close PRs),
      unless the function selects PRs through a helper;
    * `gh pr <verb> <branch name>`: acting by branch name instead of a selected number.
    """
    tree = ast.parse(source)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    offenders: set[str] = set()

    def scope(node: ast.AST) -> ast.AST:
        while node in parents and not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef
        ):
            node = parents[node]
        return node

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            tokens = _tokens([*node.args, *(k.value for k in node.keywords)])
        elif isinstance(node, ast.List | ast.Tuple):
            tokens = _tokens([node])
        else:
            continue
        verb = next(
            (
                (tokens[i + 1], tokens[i + 2] if i + 2 < len(tokens) else None)
                for i in range(len(tokens) - 1)
                if tokens[i] == "pr" and tokens[i + 1] in {"list", *ACTING}
            ),
            None,
        )
        api = "api" in tokens and (
            any(t and ("/pulls" in t or "graphql" in t) for t in tokens)
            or (
                any(t and "issues/" in t for t in tokens)
                and any(t in MUTATING for t in tokens)
            )
        )
        if verb is None and not api:
            continue
        where = f"line {node.lineno}"
        env = scope(node)
        has_helper = any(_is_helper_call(n) for n in ast.walk(env))
        if api and not has_helper:
            offenders.add(f"{where}: gh api on pulls/graphql/issues outside a helper")
        if verb is None:
            continue
        name, argument = verb
        fields = any(
            isinstance(n, ast.Name) and n.id == "OWN_FIELDS" for n in ast.walk(node)
        )
        if name == "list":
            wrapped, parent = False, node
            while parent in parents and not isinstance(parent, ast.stmt):
                parent = parents[parent]
                wrapped |= _is_helper_call(parent, WRAPPERS)
            if not wrapped and isinstance(parent, ast.Assign):
                names = {t.id for t in parent.targets if isinstance(t, ast.Name)}
                loads = [
                    n
                    for n in ast.walk(env)
                    if isinstance(n, ast.Name)
                    and n.id in names
                    and isinstance(n.ctx, ast.Load)
                ]
                wrapped = bool(loads) and all(
                    _is_helper_call(parents[n], WRAPPERS)
                    and n in getattr(parents[n], "args", [])
                    for n in loads
                )
            if isinstance(node, ast.Call):
                ok = wrapped and fields
            else:  # an argv literal: it must be built in a helper-using function
                ok = has_helper and fields
            if not ok:
                offenders.add(f"{where}: gh pr list not own_pulls+OWN_FIELDS")
        else:
            if not has_helper:
                offenders.add(f"{where}: gh pr {name} without a helper")
            if (
                argument
                and not argument.startswith("-")
                and not BY_NUMBER.match(argument)
            ):
                offenders.add(f"{where}: gh pr {name} acts on a branch name")
    return sorted(offenders)


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
        try:
            found = unguarded_pr_calls(text)
        except SyntaxError:
            continue  # not python
        offenders += [f"{path.relative_to(ROOT)} {f}" for f in found]
    assert not offenders, offenders


PLANTED = {
    "list not wrapped": 'def f():\n    return gh_json("pr", "list", "--head", "x", "--json", "number")\n',
    "list without fields": 'def f():\n    return own_pulls(gh_json("pr", "list", "--json", "number"))\n',
    "merge": 'def f(n):\n    gh("pr", "merge", str(n), "--auto")\n',
    "close in command": 'def f(n):\n    command(cwd, "gh", "pr", "close", str(n))\n',
    "subprocess list": 'def f(n):\n    subprocess.run(["gh", "pr", "merge", n])\n',
    "gh list argument": 'def f(n):\n    gh(["pr", "close", n])\n',
    "api pulls": 'def f():\n    return gh("api", "repos/{owner}/{repo}/pulls?state=open")\n',
    "api put merge": 'def f(n):\n    gh("api", "-X", "PUT", f"repos/o/r/pulls/{n}/merge")\n',
    "api graphql": 'def f():\n    gh("api", "graphql", "-f", "query=q")\n',
    "by branch name": 'def f():\n    gh("pr", "view", "refresh/x")\n',
    "list used outside own_pulls": (
        "def f():\n"
        '    x = gh_json("pr", "list", "--json", f"number,{OWN_FIELDS}")\n'
        "    own_pulls([x])\n"
        "    for pr in x:\n"
        '        gh("pr", "merge", str(pr["number"]))\n'
    ),
    "argv in a variable": (
        'def f(n):\n    args = ["gh", "pr", "merge", str(n)]\n    subprocess.run(args)\n'
    ),
    "argv as keyword": (
        'def f(n):\n    subprocess.run(args=["gh", "pr", "merge", str(n)])\n'
    ),
    "pr review": 'def f(n):\n    gh("pr", "review", str(n), "--approve")\n',
    "pr checks": 'def f(n):\n    gh("pr", "checks", str(n))\n',
    "pr diff": 'def f(n):\n    gh("pr", "diff", str(n))\n',
    "issues api patch": (
        'def f(n):\n    gh("api", "-X", "PATCH", f"repos/o/r/issues/{n}", "-f", "state=closed")\n'
    ),
}


def test_the_guard_flags_every_known_bypass():
    for name, source in PLANTED.items():
        assert unguarded_pr_calls(source), name


def test_the_guard_accepts_guarded_shapes():
    wrapped = (
        "def f():\n"
        '    pulls = own_pulls(gh_json("pr", "list", "--json", f"number,{OWN_FIELDS}"))\n'
        '    gh("pr", "merge", str(pulls[0]["number"]))\n'
    )
    assigned = (
        "def f():\n"
        '    listed = gh_json("pr", "list", "--json", f"number,{OWN_FIELDS}")\n'
        "    pulls = own_pulls(listed)\n"
        '    gh("pr", "close", str(pulls[0]["number"]))\n'
    )
    numbered = 'def f():\n    own_pulls([])\n    gh("pr", "view", "123")\n'
    for source in (wrapped, assigned, numbered):
        assert unguarded_pr_calls(source) == [], source


def _load_refresh():
    """The one loaded module: re-executing it would replace the object other tests patch."""
    from test_refresh_upstream import refresh

    return refresh


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


def test_a_fork_pr_touching_a_recipe_does_not_block_its_refresh():
    refresh = _load_refresh()
    pulls = [
        {
            "number": 5,
            "headRefName": "feature",
            "files": [{"path": "recipes/x.json"}],
            **FORK,
        },
        {
            "number": 6,
            "headRefName": "feature2",
            "files": [{"path": "recipes/y.json"}],
            **OWN,
        },
    ]
    calls: list[tuple] = []

    def gh_json(*arguments: str):
        calls.append(arguments)
        assert arguments[:2] == ("pr", "list") and OWN_FIELDS in arguments[-1]
        return pulls

    with patch("refresh_upstream.gh_json", side_effect=gh_json):
        open_prs = refresh.foreign_open_prs()
    assert open_prs == {6: {"recipes/y.json"}}  # one listing, no per-PR files calls
    footprint = ({"recipes/x.json"}, [])
    assert refresh.blocking_prs(footprint, open_prs) == []
    assert refresh.blocking_prs(({"recipes/y.json"}, []), open_prs) == [6]


class Reached(Exception):
    pass


def _publish_with_closed_pr(refresh, pr):
    assessment = refresh.Assessment("x")
    summary = refresh.Summary()
    marker = f"<!-- refresh-target: {assessment.fingerprint()} -->"
    listing = [{"number": 9, "state": "CLOSED", "body": marker, **pr}]
    with (
        patch("refresh_upstream.gh_json", return_value=listing),
        patch("refresh_upstream.gh"),
        patch("refresh_upstream.git", side_effect=Reached),
    ):
        try:
            refresh._publish(Path("."), assessment, summary, arm=False)
        except Reached:
            return summary, True
    return summary, False


def test_a_closed_fork_pr_with_the_marker_does_not_defer_a_refresh():
    refresh = _load_refresh()
    summary, proceeded = _publish_with_closed_pr(refresh, FORK)
    assert proceeded and not summary.deferred
    summary, proceeded = _publish_with_closed_pr(refresh, OWN)  # our own closed PR does
    assert not proceeded and "closed unmerged" in summary.deferred[0]


def test_an_overlapping_armed_fork_pr_does_not_defer_arming():
    refresh = _load_refresh()
    shared = [{"path": "models/shared.json"}]
    pulls = [
        {"number": 419, "autoMergeRequest": {"x": 1}, "files": shared, **FORK},
        {"number": 421, "autoMergeRequest": None, "files": shared, **OWN},
    ]
    summary = refresh.Summary()
    summary.healthy = True
    with (
        patch("refresh_upstream.gh_json", return_value=pulls),
        patch("refresh_upstream.gh") as gh,
    ):
        assert refresh.arm_mechanical_pr(421, "recipe-b", summary)
    gh.assert_called_once_with("pr", "merge", "421", "--auto", "--squash")
    assert not summary.deferred


def _big_pr(number, **extra):
    files = [{"path": f"docs/{n}.md"} for n in range(100)]
    return {
        "number": number,
        "headRefName": "feature",
        "files": files,
        "changedFiles": 130,
        **extra,
    }


def test_a_file_beyond_the_first_hundred_still_blocks_a_refresh():
    refresh = _load_refresh()
    fetched = []

    def gh_json(*arguments: str):
        assert "changedFiles" in arguments[-1]
        return [_big_pr(7, **OWN), _big_pr(8, **FORK)]

    def gh(*arguments: str, check: bool = True):
        fetched.append(arguments)
        assert "pulls/7/files" in arguments[2]  # never the fork PR's files
        return "\n".join([f"docs/{n}.md" for n in range(100)] + ["recipes/x.json"])

    with (
        patch("refresh_upstream.gh_json", side_effect=gh_json),
        patch("refresh_upstream.gh", side_effect=gh),
    ):
        open_prs = refresh.foreign_open_prs()
    assert list(open_prs) == [7] and len(fetched) == 1
    assert refresh.blocking_prs(({"recipes/x.json"}, []), open_prs) == [7]


def test_the_arming_overlap_check_sees_files_beyond_the_first_hundred():
    refresh = _load_refresh()
    armed = _big_pr(419, autoMergeRequest={"x": 1}, **OWN)
    ours = {"number": 421, "files": [{"path": "models/shared.json"}], **OWN}
    summary = refresh.Summary()
    summary.healthy = True
    with (
        patch("refresh_upstream.gh_json", return_value=[armed, ours]),
        patch(
            "refresh_upstream.gh",
            side_effect=lambda *a, **k: "models/shared.json" if a[0] == "api" else "",
        ) as gh,
    ):
        assert not refresh.arm_mechanical_pr(421, "recipe-b", summary)
    assert summary.deferred and "#419" in summary.deferred[0]
    assert not any(c.args[:2] == ("pr", "merge") for c in gh.call_args_list)
