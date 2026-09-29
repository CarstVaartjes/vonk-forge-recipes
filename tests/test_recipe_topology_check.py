"""An existing recipe may not change its Spark count; that needs a new recipe id."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/check-recipe-topology"
LOADER = importlib.machinery.SourceFileLoader("check_recipe_topology", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
topology = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = topology
LOADER.exec_module(topology)


def _recipe(slug: str, nodes: int) -> str:
    return json.dumps(
        {
            "identity": {"publisher": "vonk-forge", "slug": slug},
            "topology": {"name": "t", "node_count": nodes},
        }
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes/glm-single.json").write_text(_recipe("glm-single", 1))
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_rejects_a_changed_spark_count_and_says_to_create_a_new_recipe(
    tmp_path, capsys
) -> None:
    root = _repo(tmp_path)
    (root / "recipes/glm-single.json").write_text(_recipe("glm-single", 2))

    assert topology.main(["--root", str(root), "--base", "HEAD"]) == 1
    error = capsys.readouterr().err
    assert "glm-single" in error and "from 1 to 2" in error
    assert "new recipe id" in error


def test_rejects_the_change_even_when_the_file_is_renamed(tmp_path, capsys) -> None:
    root = _repo(tmp_path)
    (root / "recipes/glm-single.json").rename(root / "recipes/glm-renamed.json")
    (root / "recipes/glm-renamed.json").write_text(_recipe("glm-single", 2))

    assert topology.main(["--root", str(root), "--base", "HEAD"]) == 1


def test_accepts_other_changes_and_new_recipes_with_another_spark_count(
    tmp_path, capsys
) -> None:
    root = _repo(tmp_path)
    document = json.loads((root / "recipes/glm-single.json").read_text())
    document["metadata"] = {"title": "GLM"}
    (root / "recipes/glm-single.json").write_text(json.dumps(document))
    (root / "recipes/glm-dual.json").write_text(_recipe("glm-dual", 2))

    assert topology.main(["--root", str(root), "--base", "HEAD"]) == 0


def test_an_unreadable_base_fails_closed(tmp_path) -> None:
    root = _repo(tmp_path)

    assert topology.main(["--root", str(root), "--base", "no-such-ref"]) == 2
