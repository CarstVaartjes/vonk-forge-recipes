"""Authored display of a Recipe or Model, as its own small contract.

A display carries what a person or a filter reads in a catalog: titles,
description, tags, alignment, and a Model's declared facts (modalities,
capabilities, licence). It never carries execution data, and a display has no
digest of its own: the signed index that transports it is verified at ingress.

Display lives in ``display/recipes/<slug>.json``, ``display/models/<slug>.json``
and ``display/families/<slug>.json``, never in the Recipe or Model document, so the document digest covers execution
data only and a display edit never changes it. A Model fact nothing reads to
select, mount, launch or admit belongs here, so correcting it never re-runs a
proven Recipe. The catalog hierarchy is display too: a Model's ``lineage``
names its Family (product brand), Line and Model release, and a
``FamilyDisplay`` holds the titles and each Line's models in release order.
Regrouping a Model therefore never changes a digest.

Strict ``model_validate`` is for this repository's files and output;
``read_recipe_display`` / ``read_model_display`` ignore unknown fields and are
only for consumers of a published index. ``require_display_of`` is the one
identity rule: a display describes exactly the document it travels with.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, Strict, StrictStr, model_validator

from vonk_forge_contracts.capability import CapabilityName
from vonk_forge_contracts.model import (
    Description,
    ModelIdentity,
    ModelLicense,
    Slug,
    Tags,
    Title,
)
from vonk_forge_contracts.recipe import (
    Alignment,
    Identifier,
    RecipeIdentity,
    RecipeTags,
)
from vonk_forge_contracts.unrecognised import KeepUnrecognised, UnrecognisedWord

Modality = Literal["text", "image", "audio", "video", "3d", "embeddings"]
# A word a newer minor may add: a consumer reader keeps it (``UnrecognisedWord``).
Capability = Annotated[
    Annotated[CapabilityName, Strict(False)] | UnrecognisedWord, KeepUnrecognised
]
ModalityWord = Annotated[Modality | UnrecognisedWord, KeepUnrecognised]
AlignmentWord = Annotated[Alignment | UnrecognisedWord, KeepUnrecognised]


class _DisplayContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ModelDisplayIdentity(_DisplayContract):
    publisher: Slug
    slug: Slug


#: How a Model's weights relate to its release: the vendor's own released
#: checkpoint (``base``), the same weights in another format or bit-width
#: (``quantization``), or different weights such as a prune, distill or tune
#: (``derivative``).
VariantKind = Literal["base", "quantization", "derivative"]
VariantKindWord = Annotated[VariantKind | UnrecognisedWord, KeepUnrecognised]
# A Hugging Face repository id; it may name a model this catalog does not carry.
HuggingFaceId = Annotated[
    StrictStr,
    Field(
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
Facet = Annotated[StrictStr, Field(min_length=1, max_length=64)]


class ModelLineage(_DisplayContract):
    """Where a Model sits in the catalog: Family, then Line, then Model release.

    ``family``, ``line`` and ``model`` are slugs a :class:`FamilyDisplay` names
    (a Line is unique within its Family, a Model within its Line); the order
    of a Line's ``models`` there is the release order. ``generation`` is a
    facet across Lines.
    """

    family: Slug
    line: Slug
    model: Slug
    flavor: Facet | None = None
    generation: Facet | None = None
    variant_kind: VariantKindWord
    base_model: HuggingFaceId | None = None


class LineModel(_DisplayContract):
    slug: Slug
    title: Title


class LineDisplay(_DisplayContract):
    slug: Slug
    title: Title
    # The Line's Model releases, oldest first.
    models: list[LineModel] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_models(self) -> LineDisplay:
        if len({model.slug for model in self.models}) != len(self.models):
            raise ValueError("a Line names each Model once")
        return self


class FamilyDisplay(_DisplayContract):
    """One Family (product brand): its title and its Lines, in display order."""

    kind: Literal["family-display"] = "family-display"
    slug: Slug
    title: Title
    lines: list[LineDisplay] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_lines(self) -> FamilyDisplay:
        if len({line.slug for line in self.lines}) != len(self.lines):
            raise ValueError("a Family names each Line once")
        return self


class KnobDisplay(_DisplayContract):
    """What a form shows for one job knob; its type and limits stay in the recipe."""

    label: StrictStr | None = Field(default=None, min_length=1, max_length=64)
    description: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    unit: StrictStr | None = Field(default=None, min_length=1, max_length=16)


class RecipeDisplay(_DisplayContract):
    kind: Literal["recipe-display"] = "recipe-display"
    identity: RecipeIdentity
    title: Title
    description: Description
    tags: RecipeTags
    alignment: AlignmentWord | None = None
    # Wording for a job recipe's knobs, keyed by knob name (the names the
    # recipe's settings declare; the library check refuses any other).
    knobs: dict[Identifier, KnobDisplay] = Field(default_factory=dict, max_length=64)
    # The recipe's Model binding ids (``models[].id``) that are components of
    # its launch (a drafter, encoder, VAE or engram source), not what it serves;
    # their capabilities never count as served. Absent: every binding is an
    # endpoint. The library check refuses an id the recipe does not bind.
    component_bindings: list[Identifier] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def unique_component_bindings(self) -> RecipeDisplay:
        if len(self.component_bindings) != len(set(self.component_bindings)):
            raise ValueError("component_bindings must be unique")
        return self


class ModelDisplay(_DisplayContract):
    kind: Literal["model-display"] = "model-display"
    identity: ModelDisplayIdentity
    family_title: Title
    model_title: Title
    description: Description
    tags: Tags
    modalities: list[ModalityWord] = Field(min_length=1)
    # The capabilities this model supports, by name.
    capabilities: list[Capability] = Field(max_length=64)
    license: ModelLicense
    # ``None`` for a Model shown only as a helper of another (an encoder, a
    # drafter, a weight part): it has no place of its own in the hierarchy.
    lineage: ModelLineage | None = None

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
