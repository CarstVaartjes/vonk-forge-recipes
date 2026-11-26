"""Whole-library checks a producer runs over every document it publishes.

Producer-side code: the wheel keeps only the per-recipe resolution a consumer
runs on a document it fetched (``vonk_forge_contracts.resolver``).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

from vonk_forge_contracts import (
    CapabilityName,
    ModelDefinition,
    ModelDisplay,
    QualificationStatus,
    RecipeApi,
    RecipeDefinition,
    RecipeDisplay,
)
from vonk_forge_contracts.display import FamilyDisplay
from vonk_forge_contracts.recipe import RecipeJobSettings
from vonk_forge_contracts.resolver import ContractResolutionError
from vonk_forge_contracts.vocabulary import (
    CAPABILITY_TEXT,
    CATEGORY_LABEL,
    STATUS_TEXT,
    CapabilityCategory,
)


def require_complete(
    name: str, entries: Collection[object], members: Collection[object]
) -> None:
    """Raise unless ``entries`` covers exactly ``members``."""

    missing = sorted(str(m) for m in set(members) - set(entries))
    extra = sorted(str(e) for e in set(entries) - set(members))
    if missing or extra:
        raise RuntimeError(f"{name} vocabulary: missing {missing}, unknown {extra}")


def require_complete_vocabulary() -> None:
    """The published words cover every capability, status and category exactly."""

    require_complete("capability", CAPABILITY_TEXT, CapabilityName)
    require_complete("status", STATUS_TEXT, QualificationStatus)
    require_complete("category", CATEGORY_LABEL, CapabilityCategory)


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


def _check_component_bindings(recipe: RecipeDefinition, display: RecipeDisplay) -> None:
    """A display names only the recipe's bindings as components, never all of them."""

    slug = recipe.identity.slug
    bindings = {binding.id for binding in recipe.models}
    if unknown := sorted(set(display.component_bindings) - bindings):
        raise ContractResolutionError(
            f"recipe {slug}: display marks unknown binding {unknown[0]} as a component"
        )
    if bindings <= set(display.component_bindings):
        raise ContractResolutionError(
            f"recipe {slug}: display marks every binding as a component; "
            "a recipe serves at least one endpoint Model"
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
                _check_component_bindings(recipe, recipe_displays[recipe.identity.slug])
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
