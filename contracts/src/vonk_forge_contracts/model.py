"""The public model authoring contract.

The catalog stores one immutable model version/variant as one document.  The
family, logical model, and exact version are deliberately nested so a model
document is self describing when it is copied into a recipe package.

Recipes reference a model by the ``document_sha256`` of its published JSON.
"""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

_SLUG = r"^[a-z0-9][a-z0-9-]{1,62}$"
_TOKEN = r"^[a-z0-9][a-z0-9_-]{0,127}$"
_SHA256 = r"^[a-f0-9]{64}$"
_REVISION = r"^[a-f0-9]{40,64}$"


class _ModelContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Slug = Annotated[StrictStr, Field(min_length=2, max_length=63, pattern=_SLUG)]
Sha256 = Annotated[StrictStr, Field(pattern=_SHA256)]
Revision = Annotated[StrictStr, Field(pattern=_REVISION)]


def _https_or_http(value: str, field_name: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field_name} cannot contain credentials")
    return value


class ModelFamily(_ModelContract):
    publisher: Slug
    slug: Slug
    title: StrictStr = Field(min_length=1, max_length=120)


class ModelRecord(_ModelContract):
    publisher: Slug
    slug: Slug
    title: StrictStr = Field(min_length=1, max_length=120)


class ModelIdentity(_ModelContract):
    """The family, logical model, exact version, and selected variant."""

    publisher: Slug
    slug: Slug
    family: ModelFamily
    model: ModelRecord
    version: StrictStr = Field(min_length=1, max_length=128)
    variant: StrictStr = Field(min_length=1, max_length=128)


class ModelMetadata(_ModelContract):
    description: StrictStr = Field(min_length=1, max_length=4000)
    tags: list[Annotated[StrictStr, Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,39}$")]] = (
        Field(max_length=20)
    )

    @field_validator("tags")
    @classmethod
    def unique_tags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("metadata tags must be unique")
        return value


class ModelSource(_ModelContract):
    repository: StrictStr = Field(min_length=1, max_length=512)
    revision: Revision

    @field_validator("repository")
    @classmethod
    def repository_url(cls, value: str) -> str:
        return _https_or_http(value, "source.repository")


class GitHubReleaseAsset(_ModelContract):
    """One GitHub release asset selected for an existing model file."""

    file_id: StrictStr = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$"
    )
    asset_id: StrictInt = Field(gt=0)


class GitHubReleaseSource(_ModelContract):
    """An exact asset from one release in a canonical GitHub repository."""

    provider: Literal["github-release"]
    repository: StrictStr = Field(min_length=1, max_length=512)
    release_id: StrictInt = Field(gt=0)
    assets: list[GitHubReleaseAsset] = Field(min_length=1)

    @field_validator("repository")
    @classmethod
    def canonical_github_repository(cls, value: str) -> str:
        parsed = urlsplit(value)
        path = parsed.path.removeprefix("/")
        parts = path.split("/")
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or parsed.query
            or parsed.fragment
            or len(parts) != 2
            or any(
                not part
                or part in {".", ".."}
                or part.lower().endswith(".git")
                or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None
                for part in parts
            )
            or value != f"https://github.com/{parts[0]}/{parts[1]}"
        ):
            raise ValueError(
                "source.repository must be a canonical GitHub repository URL"
            )
        return value

    @field_validator("assets")
    @classmethod
    def unique_asset_locators(
        cls, value: list[GitHubReleaseAsset]
    ) -> list[GitHubReleaseAsset]:
        file_ids = [asset.file_id for asset in value]
        asset_ids = [asset.asset_id for asset in value]
        if len(file_ids) != len(set(file_ids)):
            raise ValueError("GitHub release asset file IDs must be unique")
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("GitHub release asset IDs must be unique")
        return sorted(value, key=lambda asset: asset.file_id)


def _safe_relative_path(value: str) -> str:
    if value.startswith("/") or "\\" in value or "//" in value:
        raise ValueError("file path must be relative and canonical")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("file path must not contain empty or traversal segments")
    return value


class ModelFilePart(_ModelContract):
    """One published piece of a file the source can only host split.

    A part is a transport detail: it exists only at the source (for example a
    Hugging Face repository that caps files at 50 GB publishes
    ``model.safetensors.part00``). It is never installed.
    """

    path: StrictStr = Field(min_length=1, max_length=512)
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=1)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class ModelFile(_ModelContract):
    """One entry in the complete immutable model file manifest.

    ``sha256`` and ``size_bytes`` always describe the whole installed file.
    When the source publishes the file only as split parts, ``parts`` lists
    them in joining order (byte concatenation yields the file). Omitted means
    the source publishes the file whole, so the same bytes have the same
    identity whether the source splits them or not.
    """

    id: StrictStr = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$"
    )
    path: StrictStr = Field(min_length=1, max_length=512)
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=0)
    roles: list[Annotated[StrictStr, Field(pattern=_TOKEN)]] = Field(
        min_length=1, max_length=16
    )
    # Omitted means "published whole": a document without parts serializes
    # exactly as it did before the field existed (no `"parts": null`), so
    # digests and signed plans built from a dump do not change.
    parts: list[ModelFilePart] | None = Field(
        default=None,
        min_length=2,
        max_length=1024,
        exclude_if=lambda value: value is None,
    )

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @field_validator("roles")
    @classmethod
    def unique_roles(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("file roles must be unique")
        return value

    @model_validator(mode="after")
    def empty_file_has_empty_digest(self) -> ModelFile:
        if self.size_bytes == 0 and self.sha256 != hashlib.sha256(b"").hexdigest():
            raise ValueError("zero-byte files must use the empty-content SHA-256")
        return self

    @model_validator(mode="after")
    def parts_cover_the_file(self) -> ModelFile:
        if self.parts is None:
            return self
        paths = [part.path for part in self.parts]
        if len(paths) != len(set(paths)):
            raise ValueError("file part paths must be unique")
        if self.path in paths:
            raise ValueError("a file part path must differ from the file path")
        if sum(part.size_bytes for part in self.parts) != self.size_bytes:
            raise ValueError("file part sizes must add up to the file size")
        return self


class ModelFormat(_ModelContract):
    precision: StrictStr = Field(min_length=1, max_length=64, pattern=_TOKEN)
    quantization: StrictStr = Field(min_length=1, max_length=64, pattern=_TOKEN)


class ModelReference(_ModelContract):
    kind: Literal["model"] = "model"
    publisher: StrictStr = Field(min_length=1, max_length=128)
    slug: Slug
    content_sha256: Sha256


class ModelTerritorialRestrictions(_ModelContract):
    denied_jurisdictions: list[Annotated[StrictStr, Field(pattern=r"^[A-Z]{2,3}$")]] = (
        Field(min_length=1, max_length=64)
    )
    notice: StrictStr = Field(min_length=1, max_length=1024)

    @field_validator("denied_jurisdictions")
    @classmethod
    def unique_jurisdictions(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("territorial restrictions must list unique jurisdictions")
        return value


class ModelLicense(_ModelContract):
    spdx: StrictStr = Field(min_length=1, max_length=128)
    url: StrictStr = Field(min_length=1, max_length=512)
    attribution: list[StrictStr] = Field(max_length=32)
    territorial_restrictions: ModelTerritorialRestrictions | None = None

    @field_validator("url")
    @classmethod
    def license_url(cls, value: str) -> str:
        return _https_or_http(value, "license.url")


# A capability name is data, not a closed enum: a newer catalog may declare a
# capability this contract release does not know yet, and readers keep it.
CapabilityName = Annotated[
    StrictStr, Field(min_length=1, max_length=40, pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")
]


class ModelDefinition(_ModelContract):
    """One exact model version and variant, including its complete manifest."""

    kind: Literal["model"] = "model"
    identity: ModelIdentity
    metadata: ModelMetadata
    # A provider account token is needed to download the files (a gated
    # repository). Public models download anonymously.
    requires_token: StrictBool
    dependencies: list[ModelReference] = Field(max_length=32)
    modalities: list[Literal["text", "image", "audio", "video", "3d", "embeddings"]] = (
        Field(min_length=1, max_length=6)
    )
    source: ModelSource | GitHubReleaseSource
    format: ModelFormat
    license: ModelLicense
    files: list[ModelFile] = Field(min_length=1)
    # The capabilities this model supports, by name.
    capabilities: list[CapabilityName] = Field(max_length=64)

    @model_validator(mode="after")
    def exact_snapshot(self) -> ModelDefinition:
        ids = [item.id for item in self.files]
        paths = [item.path for item in self.files]
        if len(ids) != len(set(ids)):
            raise ValueError("file IDs must be unique")
        if isinstance(self.source, GitHubReleaseSource):
            if self.requires_token:
                raise ValueError(
                    "GitHub release sources require public, anonymous model access"
                )
            if {asset.file_id for asset in self.source.assets} != set(ids):
                raise ValueError(
                    "GitHub release asset file IDs must exactly cover the model files"
                )
        if len(paths) != len(set(paths)):
            raise ValueError("file paths must be unique")
        if len(self.modalities) != len(set(self.modalities)):
            raise ValueError("modalities must be unique")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("capabilities must be unique")
        dependency_keys = [(item.publisher, item.slug) for item in self.dependencies]
        if len(dependency_keys) != len(set(dependency_keys)):
            raise ValueError("model dependencies must be unique")
        if isinstance(self.source, GitHubReleaseSource) and any(
            item.parts is not None for item in self.files
        ):
            raise ValueError(
                "GitHub release sources publish whole assets; file parts are not allowed"
            )
        part_paths = [part.path for item in self.files for part in item.parts or []]
        if len(part_paths) != len(set(part_paths)):
            raise ValueError("file part paths must be unique across the manifest")
        if set(part_paths) & set(paths):
            raise ValueError("a file part path must differ from every file path")
        sizes: dict[str, int] = {}
        for item in self.files:
            previous = sizes.setdefault(item.sha256, item.size_bytes)
            if previous != item.size_bytes:
                raise ValueError("files sharing a digest must declare the same size")
        return self

    @property
    def installed_bytes(self) -> int:
        """Total installed bytes, counting every manifest entry once."""

        return sum(item.size_bytes for item in self.files)

    @property
    def download_bytes(self) -> int:
        """Download bytes, deduplicated by content digest."""

        return sum({item.sha256: item.size_bytes for item in self.files}.values())
