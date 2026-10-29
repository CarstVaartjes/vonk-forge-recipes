"""Authored display text of a Recipe or Model, as its own small contract.

A display carries what a person reads in a catalog (titles, description, tags,
alignment). It never carries execution data, and a display has no digest of its
own: the signed index that transports it is verified at ingress.

On contracts 2.x the display is derived from the document's own ``metadata`` and
titles by ``recipe_display_of`` / ``model_display_of``. Readers use those
accessors, never the document fields.

Strict ``model_validate`` is for this repository's files and output;
``read_recipe_display`` / ``read_model_display`` ignore unknown fields and are
only for consumers of a published index. ``require_display_of`` is the one
identity rule: a display describes exactly the document it travels with.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .model import Description, ModelDefinition, ModelIdentity, Slug, Tags, Title
from .recipe import Alignment, RecipeDefinition, RecipeIdentity, RecipeTags


class _DisplayContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ModelDisplayIdentity(_DisplayContract):
    publisher: Slug
    slug: Slug


class RecipeDisplay(_DisplayContract):
    kind: Literal["recipe-display"] = "recipe-display"
    identity: RecipeIdentity
    title: Title
    description: Description
    tags: RecipeTags
    alignment: Alignment | None = None


class ModelDisplay(_DisplayContract):
    kind: Literal["model-display"] = "model-display"
    identity: ModelDisplayIdentity
    family_title: Title
    model_title: Title
    description: Description
    tags: Tags


def read_recipe_display(document: Mapping[str, object]) -> RecipeDisplay:
    """Parse a published RecipeDisplay, ignoring fields a newer minor added."""

    return RecipeDisplay.model_validate(document, extra="ignore")


def read_model_display(document: Mapping[str, object]) -> ModelDisplay:
    """Parse a published ModelDisplay, ignoring fields a newer minor added."""

    return ModelDisplay.model_validate(document, extra="ignore")


def require_display_of(
    display: RecipeDisplay | ModelDisplay,
    document_identity: RecipeIdentity | ModelIdentity,
) -> None:
    """Raise unless ``display`` names exactly the document it travels with."""

    if (display.identity.publisher, display.identity.slug) != (
        document_identity.publisher,
        document_identity.slug,
    ):
        raise ValueError("display identity differs from its document")


def published_json(document: BaseModel) -> dict[str, object]:
    """The one write form of a published contract object."""

    return document.model_dump(mode="json", exclude_unset=True)


def recipe_display_of(recipe: RecipeDefinition) -> RecipeDisplay:
    """The display of a 2.x Recipe, built from its ``metadata``."""

    metadata = recipe.metadata
    optional: dict[str, Any] = {}
    if "alignment" in metadata.model_fields_set:
        optional["alignment"] = metadata.alignment
    return RecipeDisplay(
        kind="recipe-display",
        identity=RecipeIdentity(
            publisher=recipe.identity.publisher, slug=recipe.identity.slug
        ),
        title=metadata.title,
        description=metadata.description,
        tags=list(metadata.tags),
        **optional,
    )


def model_display_of(model: ModelDefinition) -> ModelDisplay:
    """The display of a 2.x Model, built from its ``metadata`` and titles."""

    return ModelDisplay(
        kind="model-display",
        identity=ModelDisplayIdentity(
            publisher=model.identity.publisher, slug=model.identity.slug
        ),
        family_title=model.identity.family.title,
        model_title=model.identity.model.title,
        description=model.metadata.description,
        tags=list(model.metadata.tags),
    )
