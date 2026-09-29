"""Pure cross-document checks for a recipe's exact model selections.

Models are passed keyed by the ``document_sha256`` of their published JSON,
which is the digest Recipe and Model references name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .model import ModelDefinition
from .recipe import RecipeDefinition, RecipeJobServingRequest


class ContractResolutionError(ValueError):
    """The recipe does not select the supplied exact model snapshot."""


def _by_identity(
    models: Mapping[str, ModelDefinition],
) -> dict[tuple[str, str], tuple[str, ModelDefinition]]:
    result: dict[tuple[str, str], tuple[str, ModelDefinition]] = {}
    for digest, model in models.items():
        key = (model.identity.publisher, model.identity.slug)
        if key in result:
            raise ContractResolutionError(
                f"duplicate model identity: {key[0]}/{key[1]}"
            )
        result[key] = (digest, model)
    return result


def validate_model_references(models: Mapping[str, ModelDefinition]) -> None:
    """Resolve every exact dependency reference between Model documents."""

    by_identity = _by_identity(models)
    visiting: set[tuple[str, str]] = set()
    visited: set[tuple[str, str]] = set()

    def resolve(model: ModelDefinition) -> None:
        key = (model.identity.publisher, model.identity.slug)
        if key in visited:
            return
        if key in visiting:
            raise ContractResolutionError(f"model dependency cycle: {key[0]}/{key[1]}")
        visiting.add(key)
        for reference in model.dependencies:
            target_key = (reference.publisher, reference.slug)
            target = by_identity.get(target_key)
            if target is None:
                raise ContractResolutionError(
                    f"model reference is missing: {target_key[0]}/{target_key[1]}"
                )
            if target[0] != reference.content_sha256:
                raise ContractResolutionError(
                    f"model reference digest does not match: {target_key[0]}/{target_key[1]}"
                )
            resolve(target[1])
        visiting.remove(key)
        visited.add(key)

    for model in models.values():
        resolve(model)


def validate_recipe_models(
    recipe: RecipeDefinition, models: Mapping[str, ModelDefinition]
) -> None:
    """Resolve every recipe model reference and selector against model manifests."""

    by_identity = _by_identity(models)
    for selection in recipe.models:
        reference = selection.model
        key = (reference.publisher, reference.slug)
        found = by_identity.get(key)
        if found is None:
            raise ContractResolutionError(
                f"model reference is missing: {key[0]}/{key[1]}"
            )
        digest, model = found
        if digest != reference.content_sha256:
            raise ContractResolutionError(
                f"model reference digest does not match: {key[0]}/{key[1]}"
            )
        files = {item.id for item in model.files}
        for selector in selection.files:
            if selector.file_id not in files:
                raise ContractResolutionError(
                    f"selector file_id is missing from model manifest: {selector.file_id}"
                )


def validate_recipe_package_paths(
    recipe: RecipeDefinition, package_paths: Iterable[str]
) -> None:
    """Ensure build sources and filesystem job fixtures are in the package closure."""

    paths = set(package_paths)
    build = recipe.execution.build
    required_build = {
        build.context.path,
        build.dockerfile,
        *(patch.path for patch in build.patches),
    }
    missing_build = sorted(path for path in required_build if path not in paths)
    if missing_build:
        raise ContractResolutionError(
            f"build package files are missing: {', '.join(missing_build)}"
        )
    for check in recipe.validation.serving.checks:
        if not isinstance(check.request, RecipeJobServingRequest):
            continue
        required = {check.request.fixture, *check.request.input_slots.values()}
        missing = sorted(path for path in required if path not in paths)
        if missing:
            raise ContractResolutionError(
                f"job serving package files are missing: {', '.join(missing)}"
            )
