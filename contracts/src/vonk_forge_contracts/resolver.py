"""Pure cross-document checks for a recipe's exact model selections."""
from __future__ import annotations

from collections.abc import Iterable

from .canonical import content_sha256
from .model import ModelDefinition
from .recipe import RecipeDefinition, RecipeJobServingRequest


class ContractResolutionError(ValueError):
    """The recipe does not select the supplied exact model snapshot."""



def validate_model_references(models: Iterable[ModelDefinition]) -> None:
    """Resolve every exact dependency and supersedes reference in Model documents."""

    all_models = list(models)
    by_identity: dict[tuple[str, str], ModelDefinition] = {}
    for model in all_models:
        key = (model.identity.publisher, model.identity.slug)
        if key in by_identity:
            raise ContractResolutionError(f"duplicate model identity: {key[0]}/{key[1]}")
        by_identity[key] = model
    visiting: set[tuple[str, str]] = set()
    visited: set[tuple[str, str]] = set()

    def resolve(model: ModelDefinition) -> None:
        key = (model.identity.publisher, model.identity.slug)
        if key in visited:
            return
        if key in visiting:
            raise ContractResolutionError(f"model dependency cycle: {key[0]}/{key[1]}")
        visiting.add(key)
        references = [*model.dependencies, *([model.supersedes] if model.supersedes is not None else [])]
        for reference in references:
            target_key = (reference.publisher, reference.slug)
            target = by_identity.get(target_key)
            if target is None:
                raise ContractResolutionError(f"model reference is missing: {target_key[0]}/{target_key[1]}")
            if content_sha256(target) != reference.content_sha256:
                raise ContractResolutionError(f"model reference digest does not match: {target_key[0]}/{target_key[1]}")
            resolve(target)
        visiting.remove(key)
        visited.add(key)

    for model in all_models:
        resolve(model)

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
        digest = content_sha256(model)
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
