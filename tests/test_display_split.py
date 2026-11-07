"""Display text lives in its own files, so a document digest covers execution only."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
from documents import model, model_display, recipe, recipe_display
from vonk_forge_contracts import (
    CapabilityName,
    ModelDefinition,
    ModelDisplay,
    RecipeDefinition,
    RecipeDisplay,
    document_sha256,
    published_json,
)
from vonk_forge_contracts.model import ModelFamily, ModelRecord
from vonk_forge_contracts.resolver import ContractResolutionError, validate_library

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
PROVEN = json.loads((ROOT / "tests/fixtures/proven-recipe-bytes.json").read_text())


def _library() -> tuple[
    list[RecipeDefinition],
    list[ModelDefinition],
    dict[str, RecipeDisplay],
    dict[str, ModelDisplay],
]:
    recipes = sorted((ROOT / "recipes").glob("*.json"))
    models = sorted((ROOT / "models").glob("*.json"))
    return (
        [recipe(path.stem) for path in recipes],
        [model(path.stem) for path in models],
        {path.stem: recipe_display(path.stem) for path in recipes},
        {path.stem: model_display(path.stem) for path in models},
    )


def test_the_retired_display_paths_are_gone_from_the_documents() -> None:
    assert "metadata" not in RecipeDefinition.model_fields
    assert "metadata" not in ModelDefinition.model_fields
    assert "title" not in ModelFamily.model_fields
    assert "title" not in ModelRecord.model_fields
    for moved in ("modalities", "capabilities", "license"):
        assert moved not in ModelDefinition.model_fields
        assert moved in ModelDisplay.model_fields


def test_a_document_refuses_a_display_field_written_back_into_it() -> None:
    recipe_document: dict[str, Any] = published_json(recipe("d3-flash-vllm-single"))
    model_document: dict[str, Any] = published_json(
        model("deepseek-v4-1-flash-dba1be0a")
    )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RecipeDefinition.model_validate({**recipe_document, "metadata": {"title": "x"}})
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ModelDefinition.model_validate({**model_document, "metadata": {"tags": []}})
    family = {**model_document["identity"]["family"], "title": "x"}
    identity = {**model_document["identity"], "family": family}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ModelDefinition.model_validate({**model_document, "identity": identity})


def test_display_fields_are_exactly_the_moved_set() -> None:
    assert set(RecipeDisplay.model_fields) == {
        "kind",
        "identity",
        "title",
        "description",
        "tags",
        "alignment",
    }
    assert set(ModelDisplay.model_fields) == {
        "kind",
        "identity",
        "family_title",
        "model_title",
        "description",
        "tags",
        "modalities",
        "capabilities",
        "license",
    }


def test_the_library_has_exactly_one_display_per_document() -> None:
    validate_library(*_library())


def test_a_missing_display_is_refused() -> None:
    recipes, models, recipe_displays, model_displays = _library()
    victim = recipes[0].identity.slug
    del recipe_displays[victim]
    with pytest.raises(
        ContractResolutionError, match=f"recipe has no display.*{victim}"
    ):
        validate_library(recipes, models, recipe_displays, model_displays)
    recipes, models, recipe_displays, model_displays = _library()
    victim = models[0].identity.slug
    del model_displays[victim]
    with pytest.raises(
        ContractResolutionError, match=f"model has no display.*{victim}"
    ):
        validate_library(recipes, models, recipe_displays, model_displays)


def test_an_orphan_display_is_refused() -> None:
    recipes, models, recipe_displays, model_displays = _library()
    orphan = recipe_displays[recipes[0].identity.slug]
    gone = [r for r in recipes if r.identity.slug != orphan.identity.slug]
    with pytest.raises(ContractResolutionError, match="recipe display has no document"):
        validate_library(gone, models, recipe_displays, model_displays)


def test_a_duplicate_display_is_refused() -> None:
    recipes, models, recipe_displays, model_displays = _library()
    first, second = recipes[0].identity.slug, recipes[1].identity.slug
    recipe_displays[second] = recipe_displays[first]
    with pytest.raises(ContractResolutionError):
        validate_library(recipes, models, recipe_displays, model_displays)


def test_a_display_must_name_its_own_document() -> None:
    recipes, models, recipe_displays, model_displays = _library()
    first, second = recipes[0].identity.slug, recipes[1].identity.slug
    recipe_displays[first], recipe_displays[second] = (
        recipe_displays[second],
        recipe_displays[first],
    )
    with pytest.raises(ContractResolutionError, match="differs from its document"):
        validate_library(recipes, models, recipe_displays, model_displays)


def _tree_with_edited_display(tmp_path: Path, slug: str) -> Path:
    """A library whose only difference is one recipe's display file."""

    tree = tmp_path / "tree"
    tree.mkdir()
    for entry in ROOT.iterdir():
        if entry.name in {".git", "display"}:
            continue
        (tree / entry.name).symlink_to(entry)
    shutil.copytree(ROOT / "display", tree / "display")
    path = tree / "display/recipes" / f"{slug}.json"
    edited = recipe_display(slug).model_copy(
        update={
            "title": "A completely different title",
            "description": "A completely different description.",
            "tags": ["edited"],
        }
    )
    path.write_text(json.dumps(published_json(edited), indent=2, sort_keys=True) + "\n")
    return tree


def test_editing_a_proven_recipes_display_changes_no_identity_or_status(
    tmp_path: Path, built_catalog: tuple[dict, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    key = next(iter(sorted(PROVEN)))
    slug = key.split("/", 1)[1]
    tree = _tree_with_edited_display(tmp_path, slug)
    monkeypatch.setitem(TOOL["build"].__globals__, "ROOT", tree)
    # The build runs the memory gate on every recipe; it raises when it is not silent.
    edited = TOOL["build"](package_dir=tmp_path / "packages", source_commit="0" * 40)
    original, _ = built_catalog

    def projection(catalog: dict) -> list[tuple[object, ...]]:
        return [
            (
                row["source_path"],
                row["content_sha256"],
                row["source_bundle_sha256"],
                row["document"],
                row["package"],
                row["qualification"],
            )
            for row in catalog["recipes"]
        ] + [
            (row["source_path"], row["content_sha256"], row["document"])
            for row in catalog["catalog_entities"]
        ]

    assert projection(edited) == projection(original)
    shown = {
        row["document"]["identity"]["slug"]: row["display"] for row in edited["recipes"]
    }
    assert shown[slug]["title"] == "A completely different title"
    assert (
        shown[slug]
        != {
            row["document"]["identity"]["slug"]: row["display"]
            for row in original["recipes"]
        }[slug]
    )
    assert document_sha256(published_json(recipe(slug))) == next(
        row["content_sha256"]
        for row in edited["recipes"]
        if row["document"]["identity"]["slug"] == slug
    )


def test_editing_a_proven_recipes_model_facts_changes_no_identity_or_status(
    tmp_path: Path, built_catalog: tuple[dict, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The licence, capabilities and modalities of a Model are display: correcting
    one re-runs nothing."""

    key = next(iter(sorted(PROVEN)))
    model_slug = recipe(key.split("/", 1)[1]).models[0].model.slug
    tree = tmp_path / "tree"
    tree.mkdir()
    for entry in ROOT.iterdir():
        if entry.name not in {".git", "display"}:
            (tree / entry.name).symlink_to(entry)
    shutil.copytree(ROOT / "display", tree / "display")
    facts = model_display(model_slug)
    edited = facts.model_copy(
        update={
            "capabilities": [
                *facts.capabilities,
                next(c for c in CapabilityName if c not in facts.capabilities),
            ],
            "license": facts.license.model_copy(update={"spdx": "LicenseRef-edited"}),
        }
    )
    (tree / "display/models" / f"{model_slug}.json").write_text(
        json.dumps(published_json(edited), indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setitem(TOOL["build"].__globals__, "ROOT", tree)
    changed = TOOL["build"](package_dir=tmp_path / "packages", source_commit="0" * 40)
    original, _ = built_catalog
    assert [
        (r["content_sha256"], r["package"], r["qualification"])
        for r in changed["recipes"]
    ] == [
        (r["content_sha256"], r["package"], r["qualification"])
        for r in original["recipes"]
    ]
    assert [
        (e["content_sha256"], e["document"]) for e in changed["catalog_entities"]
    ] == [(e["content_sha256"], e["document"]) for e in original["catalog_entities"]]
    shown = {
        e["document"]["identity"]["slug"]: e["display"]
        for e in changed["catalog_entities"]
    }
    assert shown[model_slug]["license"]["spdx"] == "LicenseRef-edited"
