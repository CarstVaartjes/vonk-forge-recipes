"""Pure cross-document checks for a recipe's exact model selections."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .model import ModelDefinition
from .recipe import RecipeDefinition, RecipeJobServingRequest


class ContractResolutionError(ValueError):
    """The recipe does not select the supplied exact model snapshot."""


def model_content_sha256(model: ModelDefinition) -> str:
    payload = json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_recipe_models(recipe: RecipeDefinition, models: Iterable[ModelDefinition]) -> None:
    """Resolve every recipe model reference and selector against model manifests."""

    by_identity: dict[tuple[str, str], ModelDefinition] = {}
    for model in models:
        key = (model.identity.publisher, model.identity.slug)
        if key in by_identity:
            raise ContractResolutionError(f"duplicate model identity: {key[0]}/{key[1]}")
        by_identity[key] = model
    selected: dict[tuple[str, str, str], ModelDefinition] = {}
    for selection in recipe.models:
        reference = selection.model
        key = (reference.publisher, reference.slug)
        model = by_identity.get(key)
        if model is None:
            raise ContractResolutionError(f"model reference is missing: {key[0]}/{key[1]}")
        digest = model_content_sha256(model)
        if digest != reference.content_sha256:
            raise ContractResolutionError(f"model reference digest does not match: {key[0]}/{key[1]}")
        selected[(reference.publisher, reference.slug, reference.content_sha256)] = model
    for selection in recipe.models:
        reference = selection.model
        key = (reference.publisher, reference.slug, reference.content_sha256)
        model = selected.get(key)
        if model is None:
            raise ContractResolutionError(f"selector model is not selected: {selection.id}")
        files = {item.id: item for item in model.files}
        for selector in selection.files:
            file = files.get(selector.file_id)
            if file is None:
                raise ContractResolutionError(f"selector file_id is missing from model manifest: {selector.file_id}")


def validate_recipe_package_paths(recipe: RecipeDefinition, package_paths: Iterable[str]) -> None:
    """Ensure build sources and filesystem job fixtures are in the package closure."""

    paths = set(package_paths)
    if recipe.execution.mode == "build":
        build = recipe.execution.build
        required_build = {build.context.path, build.dockerfile, *(patch.path for patch in build.patches)}
        missing_build = sorted(path for path in required_build if path not in paths)
        if missing_build:
            raise ContractResolutionError(f"build package files are missing: {', '.join(missing_build)}")
    for check in recipe.validation.serving.checks:
        if not isinstance(check.request, RecipeJobServingRequest):
            continue
        required = {check.request.fixture, *check.request.input_slots.values()}
        missing = sorted(path for path in required if path not in paths)
        if missing:
            raise ContractResolutionError(f"job serving package files are missing: {', '.join(missing)}")
