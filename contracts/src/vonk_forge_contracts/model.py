"""The public model authoring contract.

The catalog stores one immutable model version/variant as one document.  The
family, logical model, and exact version are deliberately nested so a model
document is self describing when it is copied into a recipe package.
"""
from __future__ import annotations

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
    architecture: StrictStr = Field(min_length=1, max_length=128)


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
    tags: list[Annotated[StrictStr, Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,39}$")]] = Field(max_length=20)

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


class ModelFile(_ModelContract):
    """One entry in the complete immutable model file manifest."""

    id: StrictStr = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    path: StrictStr = Field(min_length=1, max_length=512)
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=0)
    roles: list[Annotated[StrictStr, Field(pattern=_TOKEN)]] = Field(min_length=1, max_length=16)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        if value.startswith("/") or "\\" in value or "//" in value:
            raise ValueError("file path must be relative and canonical")
        parts = value.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("file path must not contain empty or traversal segments")
        return value

    @field_validator("roles")
    @classmethod
    def unique_roles(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("file roles must be unique")
        return value


class ModelFormat(_ModelContract):
    container: Literal["gguf", "safetensors", "onnx", "other"]
    precision: StrictStr = Field(min_length=1, max_length=64, pattern=_TOKEN)
    quantization: StrictStr = Field(min_length=1, max_length=64, pattern=_TOKEN)


class ModelParameters(_ModelContract):
    total: StrictInt | None = Field(default=None, ge=1)
    active: StrictInt | None = Field(default=None, ge=1)


class ModelLimits(_ModelContract):
    context_tokens: StrictInt | None = Field(default=None, ge=1)
    resolution_pixels: StrictInt | None = Field(default=None, ge=1)
    frames: StrictInt | None = Field(default=None, ge=1)
    sample_rate_hz: StrictInt | None = Field(default=None, ge=1)


class ModelLicense(_ModelContract):
    spdx: StrictStr = Field(min_length=1, max_length=128)
    url: StrictStr = Field(min_length=1, max_length=512)
    attribution: list[StrictStr] = Field(max_length=32)
    operator_acceptance_required: StrictBool

    @field_validator("url")
    @classmethod
    def license_url(cls, value: str) -> str:
        return _https_or_http(value, "license.url")


CapabilityName = Literal[
    "chat", "text-generation", "text-understanding", "reasoning", "tool-use",
    "code-generation", "ocr", "image-generation", "image-understanding",
    "image-editing", "video-generation", "video-understanding", "audio-generation",
    "audio-understanding", "embeddings", "3d-generation",
]


class ModelCapabilityFact(_ModelContract):
    capability: CapabilityName
    support: Literal["supported", "unsupported", "unknown"]
    evidence_status: Literal["declared", "tested", "contradicted", "unknown"]
    evidence_digest: Sha256 | None

    @model_validator(mode="after")
    def evidence_consistency(self) -> ModelCapabilityFact:
        if self.evidence_status == "tested" and self.evidence_digest is None:
            raise ValueError("tested capability facts require an evidence digest")
        if self.evidence_status == "contradicted" and (
            self.support != "unknown" or self.evidence_digest is None
        ):
            raise ValueError("contradicted capability facts require unknown support and evidence")
        if self.evidence_status == "unknown" and self.support != "unknown":
            raise ValueError("unknown capability evidence cannot claim support")
        return self


class ModelCapabilityProvenance(_ModelContract):
    source_url: StrictStr = Field(min_length=1, max_length=512)
    source_revision: Revision
    evidence_digest: Sha256

    @field_validator("source_url")
    @classmethod
    def evidence_url(cls, value: str) -> str:
        parsed = urlsplit(_https_or_http(value, "capabilities.provenance.source_url"))
        if parsed.scheme != "https" or parsed.query or parsed.fragment:
            raise ValueError("capability evidence must use an HTTPS URL without query or fragment")
        return value


class ModelCapabilities(_ModelContract):
    schema_version: Literal[2] = 2
    facts: list[ModelCapabilityFact] = Field(max_length=64)
    provenance: ModelCapabilityProvenance

    @model_validator(mode="after")
    def facts_are_stable(self) -> ModelCapabilities:
        names = [fact.capability for fact in self.facts]
        if len(names) != len(set(names)):
            raise ValueError("capability facts must not duplicate or contradict a capability")
        self.facts = sorted(self.facts, key=lambda fact: fact.capability)
        return self


class ModelProvenance(_ModelContract):
    source_url: StrictStr = Field(min_length=1, max_length=512)
    source_revision: Revision
    evidence_digest: Sha256
    attribution: list[StrictStr] = Field(max_length=32)

    @field_validator("source_url")
    @classmethod
    def provenance_url(cls, value: str) -> str:
        return _https_or_http(value, "provenance.source_url")


class ModelDefinition(_ModelContract):
    """One exact model version and variant, including its complete manifest."""

    schema_version: Literal[2] = 2
    kind: Literal["model"] = "model"
    identity: ModelIdentity
    metadata: ModelMetadata
    modalities: list[Literal["text", "image", "audio", "video", "3d", "embeddings"]] = Field(min_length=1, max_length=6)
    source: ModelSource
    format: ModelFormat
    parameters: ModelParameters
    limits: ModelLimits
    license: ModelLicense
    files: list[ModelFile] = Field(min_length=1)
    capabilities: ModelCapabilities
    provenance: ModelProvenance

    @model_validator(mode="after")
    def exact_snapshot(self) -> ModelDefinition:
        ids = [item.id for item in self.files]
        paths = [item.path for item in self.files]
        if len(ids) != len(set(ids)):
            raise ValueError("file IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("file paths must be unique")
        if len(self.modalities) != len(set(self.modalities)):
            raise ValueError("modalities must be unique")
        self.modalities = sorted(self.modalities)
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
