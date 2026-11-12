"""The recipe library's published index, as one contract.

``catalog-index.json`` is the signed release's envelope: the Models and Recipes
with their display, API facts, qualification status and package metadata. This
module owns its shape, its schema versions and the package media type, so
producers build it through :class:`CatalogIndex` and consumers read it with
:func:`read_catalog_index`. The generated schema ships inside the release, so a
consumer can compile against exactly the data it deploys.

``published_json`` (``exclude_unset``) is the one write form: a document keeps
the bytes its ``content_sha256`` was computed from.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr
from pydantic.json_schema import models_json_schema

from .api import RecipeApi
from .display import ModelDisplay, RecipeDisplay
from .model import ModelDefinition
from .qualification_status import (
    PublishedMeasuredMemory,
    PublishedQualificationStatusView,
)
from .recipe import RecipeDefinition, RecipeRelease
from .vocabulary import CatalogVocabulary

IndexSchemaVersion = Literal[2]
PackageSchemaVersion = Literal[2]
PackageMediaType = Literal["application/vnd.vonk-forge.recipe-package.v2+tar+gzip"]

INDEX_SCHEMA_VERSION: Final[IndexSchemaVersion] = 2
INDEX_KIND: Final = "recipe-library-index"
PACKAGE_SCHEMA_VERSION: Final[PackageSchemaVersion] = 2
PACKAGE_MEDIA_TYPE: Final[PackageMediaType] = (
    "application/vnd.vonk-forge.recipe-package.v2+tar+gzip"
)
PACKAGE_PATH_PREFIX: Final = "packages/"
PACKAGE_METADATA_FIELDS: Final = (
    "path",
    "media_type",
    "sha256",
    "expected_bytes",
    "recipe_content_sha256",
    "minimum_consumer_schema",
    "source_bundle_sha256",
)

_Sha256 = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]


class _IndexContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PackageContract(_IndexContract):
    """How a consumer finds and checks the package of each recipe entry."""

    schema_version: PackageSchemaVersion
    media_type: PackageMediaType
    path_prefix: Literal["packages/"]
    metadata_fields: tuple[StrictStr, ...]


class PackageMetadata(_IndexContract):
    path: StrictStr = Field(pattern=r"^packages/[a-z0-9][a-z0-9-]*\.tar\.gz$")
    media_type: PackageMediaType
    sha256: _Sha256
    expected_bytes: int = Field(ge=1)
    recipe_content_sha256: _Sha256
    minimum_consumer_schema: int = Field(ge=1)
    source_bundle_sha256: _Sha256


class PrebuiltImage(_IndexContract):
    """The CI-built runtime image of exactly this recipe revision."""

    reference: StrictStr
    build_key: StrictStr


class CatalogModelEntry(_IndexContract):
    source_path: StrictStr
    content_sha256: _Sha256
    document: ModelDefinition
    display: ModelDisplay


class CatalogRecipeEntry(_IndexContract):
    source_path: StrictStr
    content_sha256: _Sha256
    source_bundle_sha256: _Sha256
    document: RecipeDefinition
    display: RecipeDisplay
    api: RecipeApi | None
    package: PackageMetadata
    release: RecipeRelease
    qualification: PublishedQualificationStatusView
    prebuilt_image: PrebuiltImage | None = None
    measured_memory: PublishedMeasuredMemory | None = None


class CatalogIndex(_IndexContract):
    """``catalog-index.json``: the whole published library."""

    schema_version: IndexSchemaVersion
    kind: Literal["recipe-library-index"]
    contract_version: StrictStr = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    updated_at: StrictStr = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
    )
    repository: StrictStr
    source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    package_contract: PackageContract
    vocabulary: CatalogVocabulary
    catalog_entities: tuple[CatalogModelEntry, ...]
    recipes: tuple[CatalogRecipeEntry, ...]


def read_catalog_index(document: Mapping[str, object]) -> CatalogIndex:
    """Parse a published index, ignoring fields a newer minor contract added."""

    return CatalogIndex.model_validate(document, extra="ignore")


def published_index_json(index: CatalogIndex) -> dict[str, object]:
    """The one write form of the index (``exclude_unset``, like every document)."""

    return index.model_dump(mode="json", exclude_unset=True)


def catalog_index_json_schema() -> dict[str, object]:
    """The index schema as an OpenAPI 3.1 components document.

    ``openapi-typescript`` reads it unchanged, so a consumer generates its
    types from the schema that ships in the release.
    """

    _, definitions = models_json_schema(
        [(CatalogIndex, "validation")],
        ref_template="#/components/schemas/{model}",
    )
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Vonk Forge recipe library index",
            "version": f"{INDEX_SCHEMA_VERSION}",
            "description": "catalog-index.json is a CatalogIndex.",
        },
        "paths": {},
        "components": {"schemas": definitions["$defs"]},
    }
