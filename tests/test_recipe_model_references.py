"""Every Model a recipe references must exist in this library."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from documents import model
from vonk_forge_contracts import document_sha256, published_json

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
    for name in ("models", "recipes", "display/models", "display/recipes"):
        (tmp_path / name).mkdir(parents=True)
    for slug in (MODEL, "deepseek-v4-1-flash-exl3-2-9bpw-64ba41b6"):
        shutil.copy(ROOT / "models" / f"{slug}.json", tmp_path / "models")
        shutil.copy(
            ROOT / "display/models" / f"{slug}.json", tmp_path / "display/models"
        )
    shutil.copy(ROOT / "recipes" / f"{RECIPE}.json", tmp_path / "recipes")
    shutil.copy(
        ROOT / "display/recipes" / f"{RECIPE}.json", tmp_path / "display/recipes"
    )
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
    (library / "display/models" / f"{MODEL}.json").unlink()
    [problem] = references.check(library)
    assert RECIPE in problem
    assert "deepseek-ai/deepseek-v4-1-flash-dba1be0a" in problem
    assert "no document" in problem


def test_a_stale_pinned_revision_is_named(tmp_path: Path) -> None:
    library = _library(tmp_path)
    path = library / "models" / f"{MODEL}.json"
    stale = model(MODEL).model_copy(update={"requires_token": True})
    path.write_text(json.dumps(published_json(stale)))
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


def test_a_document_without_a_display_is_named(tmp_path: Path) -> None:
    library = _library(tmp_path)
    (library / "display/recipes" / f"{RECIPE}.json").unlink()
    [problem] = references.check(library)
    assert "recipe has no display" in problem
    assert RECIPE in problem


def test_a_display_without_a_document_is_named(tmp_path: Path) -> None:
    library = _library(tmp_path)
    shutil.copy(
        ROOT / "display/recipes" / "d3-flash-vllm-single.json",
        library / "display/recipes",
    )
    [problem] = references.check(library)
    assert "recipe display has no document" in problem
