from __future__ import annotations

import json
import runpy
import tarfile
from io import BytesIO
from pathlib import Path

from vonk_forge_contracts import ModelDefinition, RecipeDefinition, content_sha256

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/canonical-synthetic-canary"
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
PACKAGE = FIXTURE / "package/canonical-synthetic-canary.tar.gz"


def _documents() -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    model = json.loads((FIXTURE / "model.json").read_text(encoding="utf-8"))
    recipe = json.loads((FIXTURE / "recipe.json").read_text(encoding="utf-8"))
    key = f"{model['identity']['publisher']}/{model['identity']['slug']}"
    return model, recipe, {key: model}


def test_canonical_canary_is_schema2_and_excluded_from_public_catalog() -> None:
    model_document, recipe_document, _ = _documents()
    model = ModelDefinition.model_validate(model_document)
    recipe = RecipeDefinition.model_validate(recipe_document)

    assert model.identity.publisher == recipe.identity.publisher == "vonk-forge-test"
    assert recipe.identity.slug == "canonical-synthetic-canary"
    assert recipe_document["models"][0]["model"]["content_sha256"] == content_sha256(model)
    assert recipe.execution.mode == "build"
    assert recipe.execution.build.base_image.platform == "linux/arm64"
    assert recipe.execution.build.base_image.digest != "0" * 64
    assert recipe.execution.build.base_image.digest != "f" * 64

    public_index = json.loads((ROOT / "catalog-index.json").read_text(encoding="utf-8"))
    public_recipes = {row["document"]["identity"]["slug"] for row in public_index["recipes"]}
    public_models = {row["document"]["identity"]["slug"] for row in public_index["catalog_entities"]}
    assert recipe.identity.slug not in public_recipes
    assert model.identity.slug not in public_models
    assert len(public_recipes) == 85
    assert len(public_models) == 92

    fixture_index = json.loads((FIXTURE / "index.json").read_text(encoding="utf-8"))
    assert fixture_index["schema_version"] == 2
    assert fixture_index["kind"] == "recipe-library-index"
    assert len(fixture_index["recipes"]) == 1
    assert len(fixture_index["catalog_entities"]) == 1
    assert fixture_index["recipes"][0]["document"]["identity"]["slug"] == recipe.identity.slug
    assert fixture_index["recipes"][0]["package"]["path"] == (
        "tests/fixtures/canonical-synthetic-canary/package/canonical-synthetic-canary.tar.gz"
    )


def test_canonical_canary_package_has_exact_source_and_model_closure() -> None:
    model_document, recipe_document, entities = _documents()
    payload, metadata = TOOL["recipe_package"](
        recipe_document,
        recipe_path=FIXTURE / "recipe.json",
        entity_documents=entities,
    )
    assert payload == PACKAGE.read_bytes()
    assert metadata["sha256"] == "41a504c098b0e26cf27408cca586f2e7323d76c9c0d06d98639c11838a2e55fd"
    TOOL["validate_recipe_archive"](payload, recipe_document, entities)

    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
        recipe_name = f"models/{model_document['identity']['slug']}.json"
        assert names == {
            "manifest.json",
            "recipe.json",
            recipe_name,
            "tests/fixtures/canonical-synthetic-canary/context/Dockerfile",
            "tests/fixtures/canonical-synthetic-canary/context/server.py",
        }
        dockerfile = archive.extractfile(
            "tests/fixtures/canonical-synthetic-canary/context/Dockerfile"
        ).read().decode()
        assert "USER 10001:10001" in dockerfile
        assert "@sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3" in dockerfile
