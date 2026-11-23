"""Pure cross-document checks for a recipe's exact model selections.

Models are passed keyed by the ``document_sha256`` of their published JSON,
which is the digest Recipe and Model references name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from vonk_forge_contracts.api import RecipeApi
from vonk_forge_contracts.display import FamilyDisplay, ModelDisplay, RecipeDisplay
from vonk_forge_contracts.model import ModelDefinition
from vonk_forge_contracts.recipe import RecipeDefinition, RecipeJobSettings


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


def _check_job_knobs(recipe: RecipeDefinition, display: RecipeDisplay) -> None:
    """Ingress rules for job knobs; a stored older recipe still parses without them."""

    slug = recipe.identity.slug
    knobs = (
        recipe.settings.knobs if isinstance(recipe.settings, RecipeJobSettings) else {}
    )
    if isinstance(recipe.settings, RecipeJobSettings):
        bound = [a.setting for a in recipe.runtime.arguments if a.setting]
        for name, knob in knobs.items():
            if bound.count(name) != 1:
                raise ContractResolutionError(
                    f"recipe {slug}: job knob {name} must be bound by exactly one "
                    "runtime argument"
                )
            if knob.change_effect != "none":
                raise ContractResolutionError(
                    f"recipe {slug}: job knob {name} must have change_effect none"
                )
    if unknown := sorted(set(display.knobs) - set(knobs)):
        raise ContractResolutionError(
            f"recipe {slug}: display describes unknown knob {unknown[0]}"
        )


def validate_library(
    recipes: Iterable[RecipeDefinition],
    models: Iterable[ModelDefinition],
    recipe_displays: Mapping[str, RecipeDisplay],
    model_displays: Mapping[str, ModelDisplay],
) -> None:
    """Require exactly one display per Recipe and Model, and no orphan display.

    Displays are keyed by their file name stem (the slug). Each document's slug
    must name a display whose identity is that document, and every display must
    belong to a document.
    """

    recipes = list(recipes)
    for label, identities, displays in (
        ("recipe", [r.identity for r in recipes], recipe_displays),
        ("model", [m.identity for m in models], model_displays),
    ):
        keys: set[tuple[str, str]] = set()
        for identity in identities:
            key = (identity.publisher, identity.slug)
            if key in keys:
                raise ContractResolutionError(
                    f"duplicate {label} identity: {key[0]}/{key[1]}"
                )
            keys.add(key)
            display = displays.get(identity.slug)
            if display is None:
                raise ContractResolutionError(
                    f"{label} has no display: {key[0]}/{key[1]}"
                )
            if (display.identity.publisher, display.identity.slug) != key:
                raise ContractResolutionError(
                    f"{label} display identity differs from its document: "
                    f"{key[0]}/{key[1]}"
                )
        if label == "recipe":
            for recipe in recipes:
                _check_job_knobs(recipe, recipe_displays[recipe.identity.slug])
        for slug, display in displays.items():
            if (display.identity.publisher, slug) not in keys:
                raise ContractResolutionError(
                    f"{label} display has no document: "
                    f"{display.identity.publisher}/{slug}"
                )


def validate_families(
    model_displays: Mapping[str, ModelDisplay],
    families: Mapping[str, FamilyDisplay],
) -> None:
    """Require every Model ``lineage`` to name a Family, Line and Model that exist,
    and every listed Model release to be named by a Model.

    Families are keyed by their file name stem (the slug); the order of a
    Line's ``models`` is its release order. Display only: a failure here never
    touches a document digest.
    """

    for slug, family in families.items():
        if family.slug != slug:
            raise ContractResolutionError(
                f"family file name differs from its slug: {slug}"
            )
    named: set[tuple[str, str, str]] = set()
    for slug, display in model_displays.items():
        lineage = display.lineage
        if lineage is None:
            continue
        family = families.get(lineage.family)
        lines = {item.slug: item for item in family.lines} if family else {}
        line = lines.get(lineage.line)
        if line is None or lineage.model not in {m.slug for m in line.models}:
            raise ContractResolutionError(
                f"model {slug}: lineage {lineage.family}/{lineage.line}/"
                f"{lineage.model} is not in display/families"
            )
        named.add((lineage.family, lineage.line, lineage.model))
    for family in families.values():
        for line in family.lines:
            for model in line.models:
                if (family.slug, line.slug, model.slug) not in named:
                    raise ContractResolutionError(
                        f"family release {family.slug}/{line.slug}/{model.slug} "
                        "has no Model"
                    )


def validate_recipe_apis(
    recipes: Iterable[RecipeDefinition], apis: Mapping[str, RecipeApi]
) -> None:
    """Reject an API document that names no recipe (a recipe may have none).

    APIs are keyed by their file name stem (the slug), like displays.
    """

    keys = {(r.identity.publisher, r.identity.slug) for r in recipes}
    for slug, api in apis.items():
        if api.identity.slug != slug:
            raise ContractResolutionError(
                f"api file name differs from its identity: {slug}"
            )
        if (api.identity.publisher, slug) not in keys:
            raise ContractResolutionError(
                f"api has no recipe: {api.identity.publisher}/{slug}"
            )


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
        missing = sorted(path for path in check.fixture_paths() if path not in paths)
        if missing:
            raise ContractResolutionError(
                f"serving package files are missing: {', '.join(missing)}"
            )
