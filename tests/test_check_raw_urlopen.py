"""New tool code that opens a provider URL by hand fails; old lines and the owner pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest
from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_raw_urlopen", str(ROOT / "tools/check-raw-urlopen")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)


@pytest.mark.parametrize(
    "code",
    [
        "with urllib.request.urlopen(request, timeout=30) as r:",
        "response = urlopen(Request(url))",
        "opener = build_opener(Handler).open(request)",
    ],
)
def test_a_raw_open_on_an_added_line_is_a_hit(code: str) -> None:
    assert guard.scan(code + "\n", {1}) == [1]


@pytest.mark.parametrize(
    "code",
    [
        "payload = fetch(request, read)",
        "# urlopen(request) is wrapped by fetch",
        "opener: Callable[..., Any] = urlopen",
    ],
)
def test_other_lines_pass(code: str) -> None:
    assert guard.scan(code + "\n", {1}) == []


def test_old_lines_pass_and_the_owner_is_exempt(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/old").write_text("urlopen(a)\nurlopen(b)\n")
    (tmp_path / "tools/_upstream_http.py").write_text("urlopen(a)\n")
    diff = (
        "+++ b/tools/old\n@@ -1,0 +2 @@\n+urlopen(b)\n"
        "+++ b/tools/_upstream_http.py\n@@ -0,0 +1 @@\n+urlopen(a)\n"
    )
    problems = guard.check(tmp_path, diff)
    assert [p.split(":")[1] for p in problems] == ["2"]


def test_a_block_moved_out_of_the_exempt_owner_is_still_new_work(
    tmp_path: Path,
) -> None:
    block = [
        "def go(url):",
        "    request = build(url)",
        "    return urlopen(request)",
        "",
    ]
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/helper.py").write_text("\n".join(block))
    gone = "".join(f"-{row}\n" for row in block)
    came = "".join(f"+{row}\n" for row in block)
    diff = (
        "diff --git a/tools/_upstream_http.py b/tools/_upstream_http.py\n"
        "--- a/tools/_upstream_http.py\n+++ b/tools/_upstream_http.py\n"
        f"@@ -1,4 +0,0 @@\n{gone}"
        "diff --git a/tools/helper.py b/tools/helper.py\n"
        "--- /dev/null\n+++ b/tools/helper.py\n"
        f"@@ -0,0 +1,4 @@\n{came}"
    )
    assert [p.split(":")[1] for p in guard.check(tmp_path, diff)] == ["3"]
