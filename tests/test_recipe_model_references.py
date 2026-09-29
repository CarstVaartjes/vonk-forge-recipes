"""Every Model a recipe references must exist in this library."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from vonk_forge_contracts import document_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/check-recipe-model-references"
LOADER = importlib.machinery.SourceFileLoader(
    "check_recipe_model_references", str(SCRIPT)
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
references = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = references
LOADER.exec_module(references)

MODEL = "deepseek-v4-1-flash-dba1be0a"
RECIPE = "deepseek-v4-1-flash-exl3-mia-dual"


def _library(tmp_path: Path) -> Path:
    for name in ("models", "recipes"):
        (tmp_path / name).mkdir()
    shutil.copy(ROOT / "models" / f"{MODEL}.json", tmp_path / "models")
    shutil.copy(
        ROOT / "models" / "deepseek-v4-1-flash-exl3-2-9bpw-64ba41b6.json",
        tmp_path / "models",
    )
    shutil.copy(ROOT / "recipes" / f"{RECIPE}.json", tmp_path / "recipes")
    return tmp_path


def test_digest_matches_the_contract_release() -> None:
    document = json.loads((ROOT / "models" / f"{MODEL}.json").read_text())
    assert references.document_sha256(document) == document_sha256(document)


def test_the_library_has_no_broken_model_reference() -> None:
    assert references.check(ROOT) == []


def test_a_complete_library_passes(tmp_path: Path) -> None:
    assert references.check(_library(tmp_path)) == []


def test_a_missing_model_document_is_named(tmp_path: Path) -> None:
    library = _library(tmp_path)
    (library / "models" / f"{MODEL}.json").unlink()
    [problem] = references.check(library)
    assert RECIPE in problem
    assert "deepseek-ai/deepseek-v4-1-flash-dba1be0a" in problem
    assert "no document" in problem


def test_a_stale_pinned_revision_is_named(tmp_path: Path) -> None:
    library = _library(tmp_path)
    path = library / "models" / f"{MODEL}.json"
    document = json.loads(path.read_text())
    document["metadata"] = {**document.get("metadata", {}), "note": "changed"}
    path.write_text(json.dumps(document))
    [problem] = references.check(library)
    assert RECIPE in problem
    assert "is pinned at" in problem


def test_a_selected_file_missing_from_the_model_is_named(tmp_path: Path) -> None:
    library = _library(tmp_path)
    path = library / "recipes" / f"{RECIPE}.json"
    document = json.loads(path.read_text())
    document["models"][1]["files"][0]["file_id"] = "not-a-file"
    path.write_text(json.dumps(document))
    problems = references.check(library)
    assert any("has no file not-a-file" in problem for problem in problems)
