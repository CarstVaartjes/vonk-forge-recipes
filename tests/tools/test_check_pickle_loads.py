"""An added pickle load fails; weights-only torch loads, creator copies and old lines pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_pickle_loads", str(ROOT / "tools/check-pickle-loads")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)


@pytest.mark.parametrize(
    "code",
    [
        "state = torch.load(path)",
        'state = torch.load(path, map_location="cpu", weights_only=False)',
        "state = torch.load(path, weights_only=flag)",
        "data = pickle.load(handle)",
        "data = pickle.loads(blob)",
        "data = _pickle.Unpickler(handle).load()",
    ],
)
def test_a_pickle_load_is_a_hit(code: str) -> None:
    assert guard.scan(code + "\n", {1}) == [1]


@pytest.mark.parametrize(
    "code",
    [
        'state = torch.load(path, map_location="cpu", weights_only=True)',
        "# state = torch.load(path)",
        'note = "torch.load(path)"',
        "blob = pickle.dumps(value)",
        "state = safetensors.torch.load_file(path)",
    ],
)
def test_weights_only_loads_and_non_loads_pass(code: str) -> None:
    assert guard.scan(code + "\n", {1}) == []


def test_an_added_line_inside_an_old_call_is_judged() -> None:
    source = "state = torch.load(\n    path,\n    weights_only=False,\n)\n"
    assert guard.scan(source, {3}) == [1]
    assert guard.scan(source, set()) == []


def test_only_added_lines_of_our_own_python_are_judged(tmp_path: Path) -> None:
    adapter = tmp_path / "adapters/a/b"
    adapter.mkdir(parents=True)
    (adapter / "vendored-upstream.json").write_text(json.dumps({"ours": ["ours.py"]}))
    (adapter / "ours.py").write_text("old = torch.load(a)\nnew = torch.load(b)\n")
    (adapter / "creator.py").write_text("x = torch.load(a)\n")
    (adapter / "NOTES.md").write_text("torch.load(a)\n")
    diff = (
        "+++ b/adapters/a/b/ours.py\n@@ -1,0 +2 @@\n+new = torch.load(b)\n"
        "+++ b/adapters/a/b/creator.py\n@@ -0,0 +1 @@\n+x = torch.load(a)\n"
        "+++ b/adapters/a/b/NOTES.md\n@@ -0,0 +1 @@\n+torch.load(a)\n"
    )
    problems = guard.check(tmp_path, diff)
    assert [p.split(":")[:2] for p in problems] == [["adapters/a/b/ours.py", "2"]]
