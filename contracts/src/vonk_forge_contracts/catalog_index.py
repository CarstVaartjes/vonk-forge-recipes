"""The recipe library's published index, as one contract.

``catalog-index.json`` is the signed release's envelope: the Models and Recipes
with their display, API facts, qualification status and package metadata. This
module owns its shape, its schema versions and the package media type, so
producers build it through :class:`CatalogIndex`. Consumers read it with
:func:`vonk_forge_contracts.catalog_reader.read_catalog`, which keeps every entry
this release can read and reports the rest; :func:`read_catalog_index` is the
strict whole-index form for producers and tests. The generated schema ships
inside the release, so a consumer can compile against exactly the data it
deploys.

``published_json`` (``exclude_unset``) is the one write form: a document keeps
the bytes its ``content_sha256`` was computed from.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    Strict,
    StrictInt,
    StrictStr,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WithJsonSchema,
    WrapValidator,
    field_validator,
)
from pydantic.json_schema import models_json_schema

from .api import RecipeApi, require_api_of
from .display import ModelDisplay, RecipeDisplay, require_display_of
from .model import ModelDefinition
from .qualification_status import (
    PublishedMeasuredMemory,
    PublishedQualificationStatusView,
)
from .recipe import RecipeDefinition, RecipeRelease
from .unrecognised import is_reading
from .vocabulary import CatalogVocabulary

IndexSchemaVersion = Literal[2]
IndexKind = Literal["recipe-library-index"]
PackagePathPrefix = Literal["packages/"]
PackageSchemaVersion = Literal[2]
PackageMediaType = Literal["application/vnd.vonk-forge.recipe-package.v2+tar+gzip"]

INDEX_SCHEMA_VERSION: Final[IndexSchemaVersion] = 2
INDEX_KIND: Final[IndexKind] = "recipe-library-index"
PACKAGE_SCHEMA_VERSION: Final[PackageSchemaVersion] = 2
PACKAGE_MEDIA_TYPE: Final[PackageMediaType] = (
    "application/vnd.vonk-forge.recipe-package.v2+tar+gzip"
)
PACKAGE_PATH_PREFIX: Final[PackagePathPrefix] = "packages/"
#: The release asset holding this contract's schema (checked in beside the models).
INDEX_SCHEMA_ASSET: Final = "catalog-index-v2.schema.json"

_Sha256 = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
# A relative POSIX path inside the repository: no leading slash, no empty, ``.``
# or ``..`` part, no backslash.
_SourcePath = Annotated[
    StrictStr,
    Field(
        max_length=512,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9._-]*(?:/[A-Za-z0-9_][A-Za-z0-9._-]*)*$",
    ),
]
#: A registry image pinned by manifest digest; a mutable tag is never a reference.
PREBUILT_REFERENCE_PATTERN: Final = (
    r"^[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]{1,5})?"
    r"(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)+"
    r"@sha256:[0-9a-f]{64}$"
)
_UPDATED_AT_PATTERN: Final = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
_UPDATED_AT_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"


def _utc_second(value: object) -> datetime:
    """A whole UTC second, written ``YYYY-MM-DDTHH:MM:SSZ``; a real calendar time."""

    if isinstance(value, datetime):
        if value.utcoffset() != timedelta(0) or value.microsecond:
            raise ValueError("updated_at must be a whole second in UTC")
        return value
    if not isinstance(value, str) or not re.fullmatch(_UPDATED_AT_PATTERN, value):
        raise ValueError("updated_at must be written YYYY-MM-DDTHH:MM:SSZ")
    # strptime refuses month 13 or second 99.
    return datetime.strptime(value, _UPDATED_AT_FORMAT).replace(tzinfo=UTC)


UtcSecond = Annotated[
    datetime,
    PlainValidator(_utc_second),
    PlainSerializer(
        lambda value: value.strftime(_UPDATED_AT_FORMAT),
        return_type=str,
        when_used="json",
    ),
    WithJsonSchema(
        {"type": "string", "format": "date-time", "pattern": _UPDATED_AT_PATTERN}
    ),
]
# A JSON array: the container accepts a list, its items stay strict.
_Array = Strict(False)


class _IndexContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PackageContract(_IndexContract):
    """How a consumer finds and checks the package of each recipe entry."""

    schema_version: PackageSchemaVersion
    media_type: PackageMediaType
    path_prefix: PackagePathPrefix
    metadata_fields: Annotated[tuple[StrictStr, ...], _Array]


class PackageMetadata(_IndexContract):
    path: StrictStr = Field(pattern=r"^packages/[a-z0-9][a-z0-9-]*\.tar\.gz$")
    media_type: PackageMediaType
    sha256: _Sha256
    expected_bytes: StrictInt = Field(ge=1)
    recipe_content_sha256: _Sha256
    minimum_consumer_schema: StrictInt = Field(ge=1)
    source_bundle_sha256: _Sha256


PACKAGE_METADATA_FIELDS: Final = tuple(PackageMetadata.model_fields)


class PrebuiltImage(_IndexContract):
    """The CI-built runtime image of exactly this recipe revision."""

    reference: StrictStr = Field(max_length=512, pattern=PREBUILT_REFERENCE_PATTERN)
    build_key: _Sha256

    @property
    def digest(self) -> str:
        """The pinned manifest digest, ``sha256:<hex>``."""

        return self.reference.rsplit("@", 1)[1]


def _dropped_by_reader(
    value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> object:
    if value is None:
        if is_reading(info):
            return None
        raise ValueError("a published entry carries this block")
    return handler(value)


def _published_form(schema: dict[str, Any]) -> None:
    # The wire always carries the block; ``None`` is the reader's drop only.
    (published,) = [item for item in schema.pop("anyOf") if item != {"type": "null"}]
    schema.update(published)


#: A block every published entry carries, which a consumer reader may drop
#: (``None``) when it cannot read it; the entry stays.
_Droppable = (
    WrapValidator(_dropped_by_reader),
    Field(json_schema_extra=_published_form),
)


class CatalogModelEntry(_IndexContract):
    source_path: _SourcePath
    content_sha256: _Sha256
    document: ModelDefinition
    # ``None`` only from a consumer reader that dropped an unreadable display.
    display: Annotated[ModelDisplay | None, *_Droppable]

    @field_validator("display")
    @classmethod
    def _display_describes_document(
        cls, display: ModelDisplay | None, info: ValidationInfo
    ) -> ModelDisplay | None:
        document = info.data.get("document")
        if display is not None and isinstance(document, ModelDefinition):
            require_display_of(display, document.identity)
        return display


class CatalogRecipeEntry(_IndexContract):
    source_path: _SourcePath
    content_sha256: _Sha256
    source_bundle_sha256: _Sha256
    document: RecipeDefinition
    # ``None`` only from a consumer reader that dropped an unreadable block.
    display: Annotated[RecipeDisplay | None, *_Droppable]
    api: RecipeApi | None
    package: PackageMetadata
    release: RecipeRelease
    qualification: Annotated[PublishedQualificationStatusView | None, *_Droppable]
    prebuilt_image: PrebuiltImage | None = None
    measured_memory: PublishedMeasuredMemory | None = None

    @field_validator("display", "api")
    @classmethod
    def _siblings_describe_document(
        cls, sibling: RecipeDisplay | RecipeApi | None, info: ValidationInfo
    ) -> RecipeDisplay | RecipeApi | None:
        document = info.data.get("document")
        if isinstance(document, RecipeDefinition):
            if isinstance(sibling, RecipeDisplay):
                require_display_of(sibling, document.identity)
            elif isinstance(sibling, RecipeApi):
                require_api_of(sibling, document.identity)
        return sibling


class CatalogIndex(_IndexContract):
    """``catalog-index.json``: the whole published library."""

    schema_version: IndexSchemaVersion
    kind: IndexKind
    contract_version: StrictStr = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    updated_at: UtcSecond
    repository: StrictStr
    source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    package_contract: PackageContract
    vocabulary: CatalogVocabulary
    catalog_entities: Annotated[tuple[CatalogModelEntry, ...], _Array]
    recipes: Annotated[tuple[CatalogRecipeEntry, ...], _Array]


def read_catalog_index(document: Mapping[str, object]) -> CatalogIndex:
    """Parse a whole index strictly: one unreadable entry refuses all of it.

    The producer and test form. A consumer reads with
    :func:`vonk_forge_contracts.catalog_reader.read_catalog`, which keeps the
    entries it can read and reports the rest.
    """

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
