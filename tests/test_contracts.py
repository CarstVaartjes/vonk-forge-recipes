from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

ROOT = Path(__file__).parents[1]

import sys

sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import (
    ModelDefinition,
    RecipeDefinition,
    content_sha256,
    model_json_schema,
    recipe_json_schema,
)
from vonk_forge_contracts.recipe import RecipeJobServingRequest
from vonk_forge_contracts.resolver import (
    validate_model_references,
    validate_recipe_models,
    validate_recipe_package_paths,
)


def load(name: str) -> dict[str, object]:
    return json.loads((ROOT / "contracts" / "src" / "vonk_forge_contracts" / "examples" / name).read_text())


def test_examples_validate_against_the_two_roots_and_generated_schemas() -> None:
    model = load("model-definition.json")
    recipe = load("recipe-image.json")
    parsed_model = ModelDefinition.model_validate(model)
    parsed_recipe = RecipeDefinition.model_validate(recipe)
    for name in ("recipe-image.json", "recipe-source-build.json", "recipe-job.json", "recipe-dual.json"):
        validate_recipe_models(RecipeDefinition.model_validate(load(name)), [parsed_model])
    Draft202012Validator(model_json_schema()).validate(model)
    Draft202012Validator(recipe_json_schema()).validate(recipe)
    assert parsed_model.kind == "model"
    assert parsed_model.modalities == ["image", "text"]
    assert [fact.capability for fact in parsed_model.capabilities.facts] == ["image-generation", "text-generation"]
    assert parsed_model.download_bytes == parsed_model.installed_bytes == 1024
    assert parsed_recipe.identity.slug == "synthetic-tiny-image"


def test_model_manifest_deduplicates_download_projection_and_rejects_conflicting_size() -> None:
    document = load("model-definition.json")
    document["files"] = [
        *document["files"],
        {
            **document["files"][0],
            "id": "config",
            "path": "config.json",
        },
    ]
    parsed = ModelDefinition.model_validate(document)
    assert parsed.installed_bytes == 2048
    assert parsed.download_bytes == 1024
    document["files"][1]["size_bytes"] = 2048
    with pytest.raises(ValidationError, match="same size"):
        ModelDefinition.model_validate(document)


def test_zero_byte_model_file_requires_the_empty_content_digest() -> None:
    document = load("model-definition.json")
    document["files"] = [{**document["files"][0], "size_bytes": 0, "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}]
    ModelDefinition.model_validate(document)
    document["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="zero-byte"):
        ModelDefinition.model_validate(document)


def test_model_selection_accepts_large_but_bounded_shard_manifests() -> None:
    recipe = load("recipe-image.json")
    recipe["models"][0]["files"] = [
        {**recipe["models"][0]["files"][0], "id": f"file-{index}", "file_id": f"file-{index}"}
        for index in range(4096)
    ]
    RecipeDefinition.model_validate(recipe)
    recipe["models"][0]["files"].append({**recipe["models"][0]["files"][0], "id": "overflow", "file_id": "overflow"})
    with pytest.raises(ValidationError, match="at most 4096"):
        RecipeDefinition.model_validate(recipe)


def test_capability_facts_are_set_semantics_and_normalized() -> None:
    document = load("model-definition.json")
    facts = document["capabilities"]["facts"]
    facts.append({"capability": "chat", "support": "unknown", "evidence_status": "unknown", "evidence_digest": None})
    facts.reverse()
    parsed = ModelDefinition.model_validate(document)
    assert [fact.capability for fact in parsed.capabilities.facts] == ["chat", "image-generation", "text-generation"]
    document["capabilities"]["facts"].append({"capability": "chat", "support": "supported", "evidence_status": "declared", "evidence_digest": None})
    with pytest.raises(ValidationError, match="duplicate or contradict"):
        ModelDefinition.model_validate(document)


def test_model_and_recipe_are_strict_and_reject_wrong_types_or_extra_fields() -> None:
    model = load("model-definition.json")
    model["schema_version"] = True
    with pytest.raises(ValidationError):
        ModelDefinition.model_validate(model)
    recipe = load("recipe-image.json")
    recipe["unexpected"] = "field"
    with pytest.raises(ValidationError):
        RecipeDefinition.model_validate(recipe)


def _build_execution() -> dict[str, object]:
    return {
        "mode": "build",
        "build": {
            "base_image": {"repository": "registry.example/vonk/base", "digest": "f" * 64, "platform": "linux/arm64"},
            "context": {"path": "context.tar"},
            "dockerfile": "Dockerfile",
            "patches": [],
            "target": None,
            "arguments": [],
            "network": {"mode": "none", "hosts": []},
        },
    }


def test_image_and_source_build_are_mutually_exclusive() -> None:
    image = load("recipe-image.json")
    build = copy.deepcopy(image)
    build["execution"] = _build_execution()
    RecipeDefinition.model_validate(build)
    image["execution"]["build"] = _build_execution()["build"]
    with pytest.raises(ValidationError):
        RecipeDefinition.model_validate(image)


def test_runtime_settings_are_checked_against_the_active_settings_variant() -> None:
    job = json.loads((ROOT / "contracts/src/vonk_forge_contracts/examples/recipe-job.json").read_text())
    job["runtime"]["arguments"] = [{"name": "context", "setting": "context_tokens"}]
    with pytest.raises(ValidationError, match="unknown setting"):
        RecipeDefinition.model_validate(job)


def test_output_cap_requires_a_positive_integer() -> None:
    image = load("recipe-image.json")
    image["validation"]["serving"]["checks"][0]["request"]["body"]["max_tokens"] = 0
    with pytest.raises(ValidationError, match="positive"):
        RecipeDefinition.model_validate(image)


def test_job_serving_request_is_filesystem_fixture_binding() -> None:
    request = RecipeJobServingRequest.model_validate(
        {"transport": "job", "fixture": "prompt", "output_path": "/outputs", "output_slot": "image"}
    )
    assert request.input_path is None
    assert request.output_path == "/outputs"
    with pytest.raises(ValidationError, match="input_path"):
        RecipeJobServingRequest.model_validate(
            {"transport": "job", "fixture": "prompt", "input_slots": {"prompt": "prompt"}, "output_path": "/outputs", "output_slot": "image"}
        )
    with pytest.raises(ValidationError):
        RecipeJobServingRequest.model_validate(
            {"transport": "job", "fixture": "../secret", "output_path": "/outputs", "output_slot": "image"}
        )


def test_job_serving_bindings_match_declared_interface_slots() -> None:
    job = load("recipe-job.json")
    request = job["validation"]["serving"]["checks"][0]["request"]
    request["output_slot"] = "missing"
    with pytest.raises(ValidationError, match="output_slot"):
        RecipeDefinition.model_validate(job)

    job = load("recipe-job.json")
    request = job["validation"]["serving"]["checks"][0]["request"]
    request.update(input_path="/inputs", input_slots={"prompt": "prompt"})
    with pytest.raises(ValidationError, match="interface input"):
        RecipeDefinition.model_validate(job)

    job["interfaces"][0]["input"] = {
        "path": "/inputs",
        "required": True,
        "media_types": ["text/plain"],
        "max_bytes": 1024,
        "slots": [
            {
                "id": "prompt",
                "label": "Prompt",
                "description": "Synthetic prompt",
                "media_types": ["text/plain"],
                "extensions": [".txt"],
                "min_files": 1,
                "max_files": 1,
                "max_file_bytes": 1024,
                "max_total_bytes": 1024,
            }
        ],
    }
    RecipeDefinition.model_validate(job)
    request["input_slots"] = {"unknown": "prompt"}
    with pytest.raises(ValidationError, match="input slot"):
        RecipeDefinition.model_validate(job)


def test_vision_checks_require_image_content_and_applicable_assertions() -> None:
    recipe = load("recipe-image.json")
    RecipeDefinition.model_validate(recipe)
    recipe["validation"]["serving"]["checks"][0]["request"]["body"]["messages"][0]["content"] = "text only"
    with pytest.raises(ValidationError, match="image_url"):
        RecipeDefinition.model_validate(recipe)

    recipe = load("recipe-image.json")
    recipe["validation"]["serving"]["checks"][0]["assertions"].append("completion.nonempty")
    with pytest.raises(ValidationError, match="applicable"):
        RecipeDefinition.model_validate(recipe)


def test_build_network_hosts_match_network_mode() -> None:
    image = load("recipe-image.json")
    image["execution"] = _build_execution()
    image["execution"]["build"]["network"]["hosts"] = ["registry.example"]
    with pytest.raises(ValidationError, match="must not declare hosts"):
        RecipeDefinition.model_validate(image)
    image["execution"]["build"]["network"] = {"mode": "public", "hosts": []}
    with pytest.raises(ValidationError, match="nonempty host allowlist"):
        RecipeDefinition.model_validate(image)


def test_job_fixture_is_required_to_be_in_the_self_contained_package() -> None:
    recipe = RecipeDefinition.model_validate(json.loads((ROOT / "contracts/src/vonk_forge_contracts/examples/recipe-job.json").read_text()))
    validate_recipe_package_paths(recipe, ["blank"])
    with pytest.raises(ValueError, match="missing"):
        validate_recipe_package_paths(recipe, [])


def test_source_build_closure_requires_context_dockerfile_and_patches() -> None:
    recipe = RecipeDefinition.model_validate(json.loads((ROOT / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json").read_text()))
    validate_recipe_package_paths(recipe, ["context.tar", "Dockerfile"])
    with pytest.raises(ValueError, match="context.tar"):
        validate_recipe_package_paths(recipe, ["Dockerfile"])


def test_pure_model_resolver_binds_identity_version_file_and_selector_digest() -> None:
    model_document = load("model-definition.json")
    model = ModelDefinition.model_validate(model_document)
    recipe_document = load("recipe-image.json")
    digest = content_sha256(model)
    recipe_document["models"][0]["model"]["content_sha256"] = digest
    recipe = RecipeDefinition.model_validate(recipe_document)
    validate_recipe_models(recipe, [model])

    for mutation in (
        lambda value: value["models"][0]["model"].update(publisher="wrong-owner"),
        lambda value: value["models"][0]["model"].update(content_sha256="f" * 64),
        lambda value: value["models"][0]["files"][0].update(file_id="missing"),
    ):
        invalid = copy.deepcopy(recipe_document)
        mutation(invalid)
        with pytest.raises((ValidationError, ValueError)):
            candidate = RecipeDefinition.model_validate(invalid)
            validate_recipe_models(candidate, [model])
    changed_model = copy.deepcopy(model_document)
    changed_model["files"][0]["sha256"] = "a" * 64
    changed = ModelDefinition.model_validate(changed_model)
    with pytest.raises(ValueError, match="digest does not match"):
        validate_recipe_models(recipe, [changed])


def test_content_digest_normalizes_defaults_and_rejects_raw_dicts() -> None:
    document = load("recipe-image.json")
    recipe = RecipeDefinition.model_validate(document)
    omitted = copy.deepcopy(document)
    omitted["settings"].pop("knobs")
    normalized = RecipeDefinition.model_validate(omitted)
    assert content_sha256(recipe) == content_sha256(normalized)
    assert content_sha256(recipe) == content_sha256(RecipeDefinition.model_validate(recipe.model_dump()))
    with pytest.raises(TypeError, match="validated"):
        content_sha256(document)  # type: ignore[arg-type]


def test_checked_in_schemas_are_generated_from_the_same_models() -> None:
    assert json.loads((ROOT / "contracts/src/vonk_forge_contracts/schema/model-definition-v2.schema.json").read_text()) == model_json_schema()
    assert json.loads((ROOT / "contracts/src/vonk_forge_contracts/schema/recipe-definition-v2.schema.json").read_text()) == recipe_json_schema()
    result = subprocess.run([sys.executable, "tools/generate-contract-schemas", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_model_access_requires_consistent_visibility_and_authentication() -> None:
    document = load("model-definition.json")
    document["access"] = {"visibility": "restricted", "gated": True, "authentication": "token"}
    ModelDefinition.model_validate(document)
    for access in (
        {"visibility": "public", "gated": True, "authentication": "none"},
        {"visibility": "restricted", "gated": False, "authentication": "token"},
        {"visibility": "public", "gated": False, "authentication": "token"},
    ):
        document["access"] = access
        with pytest.raises(ValidationError, match="must agree"):
            ModelDefinition.model_validate(document)


def test_model_references_resolve_and_reject_swapped_digest() -> None:
    source = ModelDefinition.model_validate(load("model-definition.json"))
    target_document = copy.deepcopy(source.model_dump(mode="json"))
    target_document["identity"]["slug"] = "synthetic-target"
    target_document["identity"]["model"]["slug"] = "synthetic-target"
    target_document["lineage"]["source_model"]["slug"] = "synthetic-target"
    target = ModelDefinition.model_validate(target_document)
    source_document = source.model_dump(mode="json")
    source_document["dependencies"] = [{
        "kind": "model",
        "publisher": target.identity.publisher,
        "slug": target.identity.slug,
        "content_sha256": content_sha256(target),
    }]
    source = ModelDefinition.model_validate(source_document)
    validate_model_references([source, target])
    source_document["dependencies"][0]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest does not match"):
        validate_model_references([ModelDefinition.model_validate(source_document), target])


def test_model_license_accepts_typed_territorial_restrictions() -> None:
    document = load("model-definition.json")
    document["license"]["territorial_restrictions"] = {
        "denied_jurisdictions": ["EU", "GB", "KR"],
        "notice": "This license does not apply in the listed jurisdictions.",
    }
    parsed = ModelDefinition.model_validate(document)
    assert parsed.license.territorial_restrictions is not None
    assert parsed.license.territorial_restrictions.denied_jurisdictions == ["EU", "GB", "KR"]


def test_model_license_rejects_duplicate_territories() -> None:
    document = load("model-definition.json")
    document["license"]["territorial_restrictions"] = {
        "denied_jurisdictions": ["EU", "EU"],
        "notice": "Duplicate jurisdictions are invalid.",
    }
    with pytest.raises(ValidationError, match="unique jurisdictions"):
        ModelDefinition.model_validate(document)


@pytest.mark.parametrize(
    "restrictions, message",
    [
        ({"denied_jurisdictions": ["e1"], "notice": "Invalid code."}, "string_pattern_mismatch"),
        ({"denied_jurisdictions": ["EU"], "notice": ""}, "String should have at least 1 character"),
    ],
)
def test_model_license_rejects_invalid_territorial_restrictions(
    restrictions: dict[str, object], message: str
) -> None:
    document = load("model-definition.json")
    document["license"]["territorial_restrictions"] = restrictions
    with pytest.raises(ValidationError, match=message):
        ModelDefinition.model_validate(document)
