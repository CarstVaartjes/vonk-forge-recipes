from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

ROOT = Path(__file__).parents[1]

import sys

sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import (
    GitHubReleaseSource,
    ModelDefinition,
    RecipeDefinition,
    document_sha256,
    model_json_schema,
    read_model,
    read_recipe,
    recipe_json_schema,
)
from vonk_forge_contracts.recipe import (
    MAX_RUNTIME_ARGV_TOKEN_BYTES,
    RecipeJobServingRequest,
    RecipeRuntime,
    RecipeRuntimeArgument,
    _runtime_argument_tokens,
)
from vonk_forge_contracts.resolver import (
    validate_model_references,
    validate_recipe_models,
    validate_recipe_package_paths,
)


def load(name: str) -> dict[str, Any]:
    """Load an unvalidated JSON fixture for contract mutation tests."""

    return json.loads(
        (
            ROOT / "contracts" / "src" / "vonk_forge_contracts" / "examples" / name
        ).read_text()
    )


def test_github_release_source_binds_all_model_files_and_unique_assets() -> None:
    document = cast(Any, load("model-definition.json"))
    document["files"].append(
        {**document["files"][0], "id": "second-file", "path": "second.bin"}
    )
    document["source"] = {
        "provider": "github-release",
        "repository": "https://github.com/valeoai/NAF",
        "release_id": 264676230,
        "assets": [
            {"file_id": item["id"], "asset_id": index + 1}
            for index, item in enumerate(document["files"])
        ],
    }

    model = ModelDefinition.model_validate(document)
    assert isinstance(model.source, GitHubReleaseSource)
    assert model.source.provider == "github-release"
    assert model.source.release_id == 264676230
    baseline_digest = document_sha256(document)
    round_tripped = ModelDefinition.model_validate_json(model.model_dump_json())
    assert isinstance(round_tripped.source, GitHubReleaseSource)

    changed_release = copy.deepcopy(document)
    changed_release["source"]["release_id"] += 1
    assert document_sha256(changed_release) != baseline_digest

    changed_asset = copy.deepcopy(document)
    changed_asset["source"]["assets"][0]["asset_id"] += 100
    assert document_sha256(changed_asset) != baseline_digest

    for mutation in (
        lambda source: source["assets"].pop(),
        lambda source: source["assets"][1].update(
            asset_id=source["assets"][0]["asset_id"]
        ),
        lambda source: source["assets"][0].update(asset_id=0),
        lambda source: source["assets"][0].update(file_id="missing-file"),
        lambda source: source.update(repository="https://github.com/valeoai/NAF/"),
        lambda source: source.update(
            repository="https://github.com.evil.invalid/valeoai/NAF"
        ),
    ):
        invalid = copy.deepcopy(document)
        mutation(invalid["source"])
        with pytest.raises(ValidationError):
            ModelDefinition.model_validate(invalid)

    restricted = copy.deepcopy(document)
    restricted["requires_token"] = True
    with pytest.raises(ValidationError, match="public, anonymous"):
        ModelDefinition.model_validate(restricted)


def test_examples_validate_against_the_two_roots_and_generated_schemas() -> None:
    model = load("model-definition.json")
    parsed_model = ModelDefinition.model_validate(model)
    models = {document_sha256(model): parsed_model}
    for name in ("recipe-source-build.json", "recipe-job.json", "recipe-dual.json"):
        recipe = load(name)
        validate_recipe_models(RecipeDefinition.model_validate(recipe), models)
        Draft202012Validator(recipe_json_schema()).validate(recipe)
    Draft202012Validator(model_json_schema()).validate(model)
    assert parsed_model.kind == "model"
    assert set(parsed_model.capabilities) == {"image-generation", "text-generation"}
    assert parsed_model.download_bytes == parsed_model.installed_bytes == 1024


def test_model_manifest_deduplicates_download_projection_and_rejects_conflicting_size() -> (
    None
):
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
    document["files"] = [
        {
            **document["files"][0],
            "size_bytes": 0,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
    ]
    ModelDefinition.model_validate(document)
    document["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="zero-byte"):
        ModelDefinition.model_validate(document)


def test_model_selection_accepts_large_but_bounded_shard_manifests() -> None:
    recipe = load("recipe-source-build.json")
    recipe["models"][0]["files"] = [
        {
            **recipe["models"][0]["files"][0],
            "id": f"file-{index}",
            "file_id": f"file-{index}",
        }
        for index in range(4096)
    ]
    RecipeDefinition.model_validate(recipe)
    recipe["models"][0]["files"].append(
        {**recipe["models"][0]["files"][0], "id": "overflow", "file_id": "overflow"}
    )
    with pytest.raises(ValidationError, match="at most 4096"):
        RecipeDefinition.model_validate(recipe)


def test_capabilities_are_open_unique_names() -> None:
    document = load("model-definition.json")
    document["capabilities"].append("a-capability-from-a-newer-catalog")
    ModelDefinition.model_validate(document)
    document["capabilities"].append("image-generation")
    with pytest.raises(ValidationError, match="unique"):
        ModelDefinition.model_validate(document)


def test_model_and_recipe_are_strict_and_reject_wrong_types_or_extra_fields() -> None:
    model = load("model-definition.json")
    model["requires_token"] = 0
    with pytest.raises(ValidationError):
        ModelDefinition.model_validate(model)
    recipe = load("recipe-source-build.json")
    for field in ("schema_version", "unexpected"):
        invalid = copy.deepcopy(recipe)
        invalid[field] = 2
        with pytest.raises(ValidationError):
            RecipeDefinition.model_validate(invalid)


def test_readers_ignore_fields_a_newer_minor_contract_added() -> None:
    model = load("model-definition.json")
    recipe = load("recipe-source-build.json")
    model["added_later"] = {"any": "shape"}
    recipe["topology"]["added_later"] = True
    assert read_model(model).identity.slug == "synthetic-tiny-fp16"
    assert read_recipe(recipe).identity.slug == "synthetic-tiny-build"
    with pytest.raises(ValidationError):
        RecipeDefinition.model_validate(recipe)


def test_document_digest_hashes_the_published_json_not_a_parse() -> None:
    document = load("model-definition.json")
    digest = document_sha256(document)
    reordered = dict(reversed(list(document.items())))
    assert document_sha256(reordered) == digest
    extended = {**document, "added_later": 1}
    assert read_model(extended) == read_model(document)
    assert document_sha256(extended) != digest
    with pytest.raises(TypeError):
        document_sha256(cast(Any, ModelDefinition.model_validate(document)))


def test_topology_derives_mode_fabric_and_stop_order() -> None:
    single = RecipeDefinition.model_validate(load("recipe-source-build.json"))
    dual = RecipeDefinition.model_validate(load("recipe-dual.json"))
    assert (single.topology.mode, single.topology.world_size) == ("single", 1)
    assert single.topology.fabric_connectivity == "none"
    assert single.topology.fabric_minimum_bandwidth_mbps == 0
    assert (dual.topology.mode, dual.topology.world_size) == ("distributed", 2)
    assert dual.topology.fabric_connectivity == "connected"
    assert dual.topology.fabric_minimum_bandwidth_mbps > 0
    assert dual.topology.start_order == ["worker", "entrypoint"]
    assert dual.topology.stop_order == ["entrypoint", "worker"]


def test_runtime_settings_are_checked_against_the_active_settings_variant() -> None:
    job = json.loads(
        (
            ROOT / "contracts/src/vonk_forge_contracts/examples/recipe-job.json"
        ).read_text()
    )
    job["runtime"]["arguments"] = [{"name": "context", "setting": "context_tokens"}]
    with pytest.raises(ValidationError, match="unknown setting"):
        RecipeDefinition.model_validate(job)


def test_runtime_arguments_preserve_unfamiliar_names_values_and_order() -> None:
    arguments = [
        {"name": "unknown-option", "value": "value with spaces; $HOME/Δ and {json}"},
        {"name": "unknown-option", "value": "second occurrence"},
        {
            "name": "structured_option",
            "value": {"enabled": True, "items": ["a", 3, 0.25]},
        },
        {"name": "another-option", "value": [False, {"nested": "unchanged"}]},
    ]
    parsed = [RecipeRuntimeArgument.model_validate(argument) for argument in arguments]
    assert [argument.name for argument in parsed] == [
        argument["name"] for argument in arguments
    ]
    assert [argument.model_dump(mode="json") for argument in parsed] == [
        {**argument, "setting": None} for argument in arguments
    ]


def test_runtime_arguments_reject_nul_nonfinite_and_unbounded_values() -> None:
    with pytest.raises(ValidationError, match="NUL"):
        RecipeRuntimeArgument.model_validate({"name": "--bad\x00name", "value": "ok"})
    for name in ("unknown option", "unknown.option", "--unknown"):
        with pytest.raises(ValidationError):
            RecipeRuntimeArgument.model_validate({"name": name, "value": "ok"})
    with pytest.raises(ValidationError, match="NUL"):
        RecipeRuntimeArgument.model_validate(
            {"name": "--bad", "value": {"key": "bad\x00value"}}
        )
    with pytest.raises(ValidationError, match="finite"):
        RecipeRuntimeArgument.model_validate({"name": "--bad", "value": float("inf")})
    with pytest.raises(ValidationError, match="maximum nesting"):
        RecipeRuntimeArgument.model_validate(
            {"name": "--bad", "value": [[[[[[[[["deep"]]]]]]]]]}
        )
    with pytest.raises(ValidationError, match="maximum UTF-8 size"):
        RecipeRuntimeArgument.model_validate(
            {"name": "--bad", "value": ["x" * 4096] * 20}
        )


def test_runtime_argument_null_placeholder_and_boolean_or_empty_rendering() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RecipeRuntimeArgument.model_validate({"name": "literal-null", "value": None})
    setting = RecipeRuntimeArgument.model_validate(
        {"name": "context", "value": None, "setting": "context_tokens"}
    )
    assert _runtime_argument_tokens(setting) == ["--context"]
    assert _runtime_argument_tokens(
        RecipeRuntimeArgument.model_validate({"name": "enabled", "value": True})
    ) == ["--enabled"]
    assert (
        _runtime_argument_tokens(
            RecipeRuntimeArgument.model_validate({"name": "disabled", "value": False})
        )
        == []
    )
    assert _runtime_argument_tokens(
        RecipeRuntimeArgument.model_validate({"name": "empty", "value": ""})
    ) == ["--empty", ""]
    assert _runtime_argument_tokens(
        RecipeRuntimeArgument.model_validate(
            {"name": "json", "value": {"z": 1, "a": "x"}}
        )
    ) == ["--json", '{"a":"x","z":1}']


def test_runtime_argument_utf8_and_rendered_argv_boundaries() -> None:
    exact = "é" * (MAX_RUNTIME_ARGV_TOKEN_BYTES // len("é".encode()))
    assert len(exact.encode("utf-8")) == MAX_RUNTIME_ARGV_TOKEN_BYTES
    assert (
        RecipeRuntimeArgument.model_validate({"name": "utf8", "value": exact}).value
        == exact
    )
    with pytest.raises(ValidationError, match="maximum UTF-8 size"):
        RecipeRuntimeArgument.model_validate({"name": "utf8", "value": exact + "é"})

    arguments = [
        {"name": f"option-{index}", "value": "x" * MAX_RUNTIME_ARGV_TOKEN_BYTES}
        for index in range(15)
    ]
    runtime = RecipeRuntime.model_validate(
        {
            "engine": "engine",
            "entrypoint": ["launcher"],
            "arguments": arguments,
            "environment": [],
            "lifecycle": {"stop_timeout_seconds": 1},
        }
    )
    assert len(runtime.arguments) == 15
    with pytest.raises(ValidationError, match="maximum rendered size"):
        RecipeRuntime.model_validate(
            {
                "engine": "engine",
                "entrypoint": ["launcher"],
                "arguments": [
                    *arguments,
                    {"name": "last", "value": "x" * MAX_RUNTIME_ARGV_TOKEN_BYTES},
                ],
                "environment": [],
                "lifecycle": {"stop_timeout_seconds": 1},
            }
        )


def test_entrypoint_allows_shell_punctuation_and_rejects_nul() -> None:
    runtime = {
        "engine": "engine",
        "entrypoint": ["launcher", "value with spaces; $HOME/Δ", '{"json": true}'],
        "arguments": [],
        "environment": [],
        "lifecycle": {"stop_timeout_seconds": 1},
    }
    assert RecipeRuntime.model_validate(runtime).entrypoint[1] == (
        "value with spaces; $HOME/Δ"
    )
    runtime["entrypoint"] = ["launcher", "bad\x00value"]
    with pytest.raises(ValidationError, match="NUL"):
        RecipeRuntime.model_validate(runtime)


def test_output_cap_requires_a_positive_integer() -> None:
    recipe = load("recipe-source-build.json")
    recipe["validation"]["serving"]["checks"][0]["request"]["body"]["max_tokens"] = 0
    with pytest.raises(ValidationError, match="positive"):
        RecipeDefinition.model_validate(recipe)


def test_job_serving_request_is_filesystem_fixture_binding() -> None:
    request = RecipeJobServingRequest.model_validate(
        {"transport": "job", "fixture": "prompt", "output_slot": "image"}
    )
    assert request.input_slots == {}
    for invalid in (
        {"fixture": "../secret"},
        {"fixture": "prompt", "output_path": "/outputs"},
    ):
        with pytest.raises(ValidationError):
            RecipeJobServingRequest.model_validate(
                {"transport": "job", "output_slot": "image", **invalid}
            )


def test_job_serving_bindings_match_declared_interface_slots() -> None:
    job = load("recipe-job.json")
    request = job["validation"]["serving"]["checks"][0]["request"]
    request["output_slot"] = "missing"
    with pytest.raises(ValidationError, match="output_slot"):
        RecipeDefinition.model_validate(job)

    job = load("recipe-job.json")
    request = job["validation"]["serving"]["checks"][0]["request"]
    request.update(input_slots={"prompt": "prompt"})
    with pytest.raises(ValidationError, match="interface input"):
        RecipeDefinition.model_validate(job)

    job["interfaces"][0]["input"] = {
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
    recipe = load("recipe-source-build.json")
    RecipeDefinition.model_validate(recipe)
    recipe["validation"]["serving"]["checks"][0]["request"]["body"]["messages"][0][
        "content"
    ] = "text only"
    with pytest.raises(ValidationError, match="image_url"):
        RecipeDefinition.model_validate(recipe)

    recipe = load("recipe-source-build.json")
    recipe["validation"]["serving"]["checks"][0]["assertions"].append(
        "completion.nonempty"
    )
    with pytest.raises(ValidationError, match="applicable"):
        RecipeDefinition.model_validate(recipe)


def test_build_network_hosts_are_unique_and_empty_means_offline() -> None:
    recipe = load("recipe-source-build.json")
    network = recipe["execution"]["build"]["network"]
    assert network["hosts"] == []
    network["hosts"] = ["registry.example"]
    RecipeDefinition.model_validate(recipe)
    network["hosts"] = ["registry.example", "registry.example"]
    with pytest.raises(ValidationError, match="unique"):
        RecipeDefinition.model_validate(recipe)
    network.update(hosts=[], mode="none")
    with pytest.raises(ValidationError):
        RecipeDefinition.model_validate(recipe)


def test_job_fixture_is_required_to_be_in_the_self_contained_package() -> None:
    recipe = RecipeDefinition.model_validate(
        json.loads(
            (
                ROOT / "contracts/src/vonk_forge_contracts/examples/recipe-job.json"
            ).read_text()
        )
    )
    validate_recipe_package_paths(recipe, ["blank", "context.tar", "Dockerfile"])
    with pytest.raises(ValueError, match="blank"):
        validate_recipe_package_paths(recipe, ["context.tar", "Dockerfile"])


def test_source_build_closure_requires_context_dockerfile_and_patches() -> None:
    recipe = RecipeDefinition.model_validate(
        json.loads(
            (
                ROOT
                / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json"
            ).read_text()
        )
    )
    validate_recipe_package_paths(recipe, ["context.tar", "Dockerfile"])
    with pytest.raises(ValueError, match="context.tar"):
        validate_recipe_package_paths(recipe, ["Dockerfile"])


def test_pure_model_resolver_binds_identity_version_file_and_selector_digest() -> None:
    model_document = load("model-definition.json")
    model = ModelDefinition.model_validate(model_document)
    recipe_document = load("recipe-source-build.json")
    digest = document_sha256(model_document)
    recipe_document["models"][0]["model"]["content_sha256"] = digest
    recipe = RecipeDefinition.model_validate(recipe_document)
    validate_recipe_models(recipe, {digest: model})

    for mutation in (
        lambda value: value["models"][0]["model"].update(publisher="wrong-owner"),
        lambda value: value["models"][0]["model"].update(content_sha256="f" * 64),
        lambda value: value["models"][0]["files"][0].update(file_id="missing"),
    ):
        invalid = copy.deepcopy(recipe_document)
        mutation(invalid)
        with pytest.raises((ValidationError, ValueError)):
            candidate = RecipeDefinition.model_validate(invalid)
            validate_recipe_models(candidate, {digest: model})
    changed_model = copy.deepcopy(model_document)
    changed_model["files"][0]["sha256"] = "a" * 64
    changed = ModelDefinition.model_validate(changed_model)
    with pytest.raises(ValueError, match="digest does not match"):
        validate_recipe_models(recipe, {document_sha256(changed_model): changed})


def test_checked_in_schemas_are_generated_from_the_same_models() -> None:
    assert (
        json.loads(
            (
                ROOT
                / "contracts/src/vonk_forge_contracts/schema/model-definition-v2.schema.json"
            ).read_text()
        )
        == model_json_schema()
    )
    assert (
        json.loads(
            (
                ROOT
                / "contracts/src/vonk_forge_contracts/schema/recipe-definition-v2.schema.json"
            ).read_text()
        )
        == recipe_json_schema()
    )
    result = subprocess.run(
        [sys.executable, "tools/generate-contract-schemas", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_model_references_resolve_and_reject_swapped_digest() -> None:
    target_document = load("model-definition.json")
    target_document["identity"]["slug"] = "synthetic-target"
    target_document["identity"]["model"]["slug"] = "synthetic-target"
    target_digest = document_sha256(target_document)
    target = ModelDefinition.model_validate(target_document)
    source_document = load("model-definition.json")
    source_document["dependencies"] = [
        {
            "kind": "model",
            "publisher": target.identity.publisher,
            "slug": target.identity.slug,
            "content_sha256": target_digest,
        }
    ]

    def models() -> dict[str, ModelDefinition]:
        source = ModelDefinition.model_validate(source_document)
        return {document_sha256(source_document): source, target_digest: target}

    validate_model_references(models())
    source_document["dependencies"][0]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest does not match"):
        validate_model_references(models())


def test_model_license_accepts_typed_territorial_restrictions() -> None:
    document = load("model-definition.json")
    document["license"]["territorial_restrictions"] = {
        "denied_jurisdictions": ["EU", "GB", "KR"],
        "notice": "This license does not apply in the listed jurisdictions.",
    }
    parsed = ModelDefinition.model_validate(document)
    assert parsed.license.territorial_restrictions is not None
    assert parsed.license.territorial_restrictions.denied_jurisdictions == [
        "EU",
        "GB",
        "KR",
    ]


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
        (
            {"denied_jurisdictions": ["e1"], "notice": "Invalid code."},
            "string_pattern_mismatch",
        ),
        (
            {"denied_jurisdictions": ["EU"], "notice": ""},
            "String should have at least 1 character",
        ),
    ],
)
def test_model_license_rejects_invalid_territorial_restrictions(
    restrictions: dict[str, object], message: str
) -> None:
    document = load("model-definition.json")
    document["license"]["territorial_restrictions"] = restrictions
    with pytest.raises(ValidationError, match=message):
        ModelDefinition.model_validate(document)
