"""Authored display text of a Recipe or Model, as its own small contract.

A display carries what a person reads in a catalog (titles, description, tags,
alignment). It never carries execution data, and a display has no digest of its
own: the signed index that transports it is verified at ingress.

On contracts 2.x the display is derived from the document's own ``metadata`` and
titles by ``recipe_display_of`` / ``model_display_of``. Readers use those
accessors, never the document fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
)

from .model import ModelDefinition, Slug
from .recipe import RecipeDefinition, RecipeIdentity

Tag = Annotated[StrictStr, Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,39}$")]


class _DisplayContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ModelDisplayIdentity(_DisplayContract):
    publisher: Slug
    slug: Slug


class RecipeDisplay(_DisplayContract):
    kind: Literal["recipe-display"] = "recipe-display"
    identity: RecipeIdentity
    title: StrictStr = Field(min_length=1, max_length=120)
    description: StrictStr = Field(min_length=1, max_length=4000)
    tags: list[StrictStr] = Field(max_length=20)
    alignment: (
        Literal["standard", "abliterated", "derisked", "other-modified", "unspecified"]
        | None
    ) = None


class ModelDisplay(_DisplayContract):
    kind: Literal["model-display"] = "model-display"
    identity: ModelDisplayIdentity
    family_title: StrictStr = Field(min_length=1, max_length=120)
    model_title: StrictStr = Field(min_length=1, max_length=120)
    description: StrictStr = Field(min_length=1, max_length=4000)
    tags: list[Tag] = Field(max_length=20)

    @field_validator("tags")
    @classmethod
    def unique_tags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("display tags must be unique")
        return value


def read_recipe_display(document: Mapping[str, object]) -> RecipeDisplay:
    """Parse a published RecipeDisplay, ignoring fields a newer minor added."""

    return RecipeDisplay.model_validate(document, extra="ignore")


def read_model_display(document: Mapping[str, object]) -> ModelDisplay:
    """Parse a published ModelDisplay, ignoring fields a newer minor added."""

    return ModelDisplay.model_validate(document, extra="ignore")


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
