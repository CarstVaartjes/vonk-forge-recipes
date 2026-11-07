"""Authored display of a Recipe or Model, as its own small contract.

A display carries what a person or a filter reads in a catalog: titles,
description, tags, alignment, and a Model's declared facts (modalities,
capabilities, licence). It never carries execution data, and a display has no
digest of its own: the signed index that transports it is verified at ingress.

Display lives in ``display/recipes/<slug>.json`` and ``display/models/<slug>.json``,
never in the Recipe or Model document, so the document digest covers execution
data only and a display edit never changes it. A Model fact nothing reads to
select, mount, launch or admit belongs here, so correcting it never re-runs a
proven Recipe.

Strict ``model_validate`` is for this repository's files and output;
``read_recipe_display`` / ``read_model_display`` ignore unknown fields and are
only for consumers of a published index. ``require_display_of`` is the one
identity rule: a display describes exactly the document it travels with.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .capability import CapabilityName
from .model import (
    Description,
    ModelIdentity,
    ModelLicense,
    Slug,
    Tags,
    Title,
)
from .recipe import Alignment, RecipeIdentity, RecipeTags


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
    modalities: list[Literal["text", "image", "audio", "video", "3d", "embeddings"]] = (
        Field(min_length=1, max_length=6)
    )
    # The capabilities this model supports, by name.
    capabilities: list[Annotated[CapabilityName, Field(strict=False)]] = Field(
        max_length=64
    )
    license: ModelLicense

    @model_validator(mode="after")
    def unique_facts(self) -> ModelDisplay:
        if len(self.modalities) != len(set(self.modalities)):
            raise ValueError("modalities must be unique")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("capabilities must be unique")
        return self


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
