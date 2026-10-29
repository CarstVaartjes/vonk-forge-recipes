"""New raw reads of Recipe, Model or display JSON fail; contract reads and old lines pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_document_reads", str(ROOT / "tools/check-document-reads")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

RAW_RECIPE = """
import json
def f(s):
    return json.loads((ROOT / "recipes" / f"{s}.json").read_text())["runtime"]
"""
RAW_MODEL = """
import json
def f(base):
    p = base / "models" / "m.json"
    doc = json.loads(p.read_text())
    return doc["identity"]
"""
RAW_PARAMETER = """
from typing import Any
def f(document: dict[str, Any]):
    return document.get("metadata")
"""
RAW_DISPLAY = """
import json
def f():
    return json.load(open("display/recipes/x.json"))
"""
CONTRACT = """
import json
from vonk_forge_contracts import read_recipe
def f(path, recipe):
    parsed = read_recipe(json.loads(path.read_text()))
    return parsed.runtime.engine, recipe.runtime.engine
"""
UNRELATED = """
def f(entry, roles):
    return entry["status"], roles != ["metadata"], {"a": 1}.get("title")
"""
OLD_RAW = "import json\nx = json.loads(open('recipes/a.json').read())['runtime']\n"


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _check(tmp_path: Path, files: dict[str, str], old: dict[str, str] | None = None):
    _git(tmp_path, "init", "-q", "-b", "main")
    for name, text in (old or {"README": "x\n"}).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "branch", "base")
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(textwrap.dedent(text))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "head")
    return guard.check(tmp_path, "base")


def test_added_raw_recipe_read_fails(tmp_path: Path) -> None:
    problems = _check(tmp_path, {"tools/new.py": RAW_RECIPE})
    assert [p.split(":")[1] for p in problems] == ["4", "4"]
    assert "raw load read" in problems[0] and "raw subscript read" in problems[1]


def test_added_raw_model_read_fails(tmp_path: Path) -> None:
    problems = _check(tmp_path, {"spark_sweep/new.py": RAW_MODEL})
    assert any(":5: raw load" in p for p in problems)
    assert any(":6: raw subscript" in p for p in problems)


def test_annotated_document_parameter_fails(tmp_path: Path) -> None:
    problems = _check(tmp_path, {"scripts/new.py": RAW_PARAMETER})
    assert [p.split(":")[1] for p in problems] == ["4"]


def test_added_display_read_fails(tmp_path: Path) -> None:
    problems = _check(tmp_path, {"tests/test_new.py": RAW_DISPLAY})
    assert problems and all(":4: raw load" in p for p in problems)


def test_extensionless_python_tool_is_checked(tmp_path: Path) -> None:
    problems = _check(
        tmp_path, {"tools/new-tool": "#!/usr/bin/env python3" + RAW_RECIPE}
    )
    assert problems


def test_contract_reads_and_unrelated_subscripts_pass(tmp_path: Path) -> None:
    files = {"tools/ok.py": CONTRACT, "tools/other.py": UNRELATED}
    assert _check(tmp_path, files) == []


def test_exempt_paths_pass(tmp_path: Path) -> None:
    files = {
        "qualification/documents.py": RAW_RECIPE,
        "tests/documents.py": RAW_MODEL,
        "adapters/x/a.py": RAW_RECIPE,
        "contracts/src/vonk_forge_contracts/r.py": RAW_RECIPE,
    }
    assert _check(tmp_path, files) == []


def test_unchanged_raw_reads_pass_and_only_added_lines_count(tmp_path: Path) -> None:
    old = {"tools/old.py": OLD_RAW}
    assert _check(tmp_path, {"tools/old.py": OLD_RAW + "y = 1\n"}, old) == []


def test_cli_exits_one_on_a_hit(tmp_path: Path) -> None:
    _check(tmp_path, {"tools/new.py": RAW_RECIPE})
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check-document-reads"),
            "--base",
            "base",
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1 and "tools/new.py:4" in run.stdout
