"""The wheel stays wire: an added producer-side function or filesystem walk fails."""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from added_diff import added_diff
from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_wire_only", str(ROOT / "tools/check-wire-only")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

PACKAGE = "contracts/src/vonk_forge_contracts"
INIT = 'SHARED_FUNCTIONS: Final = ("read_model",)\n'
RESOLVER = "def read_model(text):\n    return text\n"
EVERYTHING = "def validate_everything(root: Path):\n    return list(root.glob('*'))\n"


def library(tmp_path: Path, resolver: str) -> Path:
    (tmp_path / PACKAGE).mkdir(parents=True)
    (tmp_path / PACKAGE / "__init__.py").write_text(INIT)
    (tmp_path / PACKAGE / "resolver.py").write_text(resolver)
    return tmp_path


def run(root: Path, diff: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    diff_file = tmp_path / "added.diff"
    diff_file.write_text(diff)
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check-wire-only"),
            "--root",
            str(root),
            "--diff",
            str(diff_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_an_added_public_function_outside_the_owner_list_exits_one(
    tmp_path: Path,
) -> None:
    source = RESOLVER + "\n\n" + EVERYTHING
    root = library(tmp_path / "head", source)
    diff = added_diff(
        {f"{PACKAGE}/resolver.py": source}, old_lines={f"{PACKAGE}/resolver.py": 3}
    )
    result = run(root, diff, tmp_path)
    assert result.returncode == 1
    assert {n for n, _ in guard.scan(source, {5, 6}, {"read_model"})} == {5, 6}


@pytest.mark.parametrize(
    "line",
    [
        "    names = path.rglob('*.json')",
        "    return list(path.iterdir())",
        "    return list(os.walk(path))",
        "    return glob('*.json')",
        "    return open(path).read()",
        "    return path.read_text()",
        "    return path.read_bytes()",
    ],
)
def test_filesystem_traversal_in_a_listed_function_fails(line: str) -> None:
    source = f"def read_model(path):\n{line}\n"
    assert [n for n, _ in guard.scan(source, {2}, {"read_model"})] == [2]


def test_a_library_root_parameter_fails() -> None:
    source = "def read_model(library_root):\n    return library_root\n"
    assert [n for n, _ in guard.scan(source, {1}, {"read_model"})] == [1]


def test_a_listed_function_a_private_helper_and_old_lines_pass() -> None:
    source = (
        RESOLVER
        + "\n\ndef _helper(path):\n    return path\n\n\ndef other(path):\n    return path\n"
    )
    assert guard.scan(source, {1, 2, 5, 6}, {"read_model"}) == []
    assert [n for n, _ in guard.scan(source, {9}, {"read_model"})] == [9]


def test_a_line_outside_the_wheel_is_not_judged(tmp_path: Path) -> None:
    diff = added_diff({"src/recipe_tooling/other.py": EVERYTHING})
    assert guard.check(library(tmp_path, RESOLVER), diff) == []


def test_the_owner_list_is_exactly_the_wheels_public_functions() -> None:
    names: set[str] = set()
    for path in (ROOT / PACKAGE).glob("*.py"):
        names |= {
            node.name
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        }
    owner = (ROOT / PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert guard.shared_functions(owner) == names
