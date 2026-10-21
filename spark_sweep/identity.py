"""Persisted identity of accepted work, independent of mutable catalog paging."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict
from vonk_forge_contracts import QualificationIdentity, QualificationRuntimeImage

from .catalog import Recipe
from .catalog_contract import RuntimeImageIdentity


class VerdictIdentity(BaseModel):
    """Content certified by one verdict; absent image evidence stays unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_sha256: str
    runtime_image: RuntimeImageIdentity | None = None
    model_digests: tuple[str, ...]
    # Unknown image pins bind accepted work conservatively to its library for
    # cache/request matching. Qualification decisions use evidence, not this key.
    # Never substitute the Docker base image.
    image_library_commit: str | None = None


def content_identity(recipe: Recipe) -> VerdictIdentity:
    return VerdictIdentity(
        content_sha256=recipe.content_sha256,
        runtime_image=recipe.runtime_image,
        model_digests=tuple(sorted(set(recipe.model_digests))),
        image_library_commit=recipe.library_commit
        if recipe.runtime_image is None
        else None,
    )


def current_identity(recipe: Recipe) -> QualificationIdentity:
    identity = content_identity(recipe)
    return QualificationIdentity(
        content_sha256=identity.content_sha256,
        runtime_image=QualificationRuntimeImage.model_validate(
            identity.runtime_image.model_dump()
        )
        if identity.runtime_image is not None
        else None,
        model_digests=identity.model_digests,
        required_sparks=recipe.node_count,
    )


def recorded_identity(entry: Mapping[str, Any]) -> VerdictIdentity | None:
    value = entry.get("verdict_identity")
    return VerdictIdentity.model_validate(value) if value is not None else None


def verdict_matches(entry: Mapping[str, Any], recipe: Recipe) -> bool:
    identity = content_identity(recipe)
    recorded = recorded_identity(entry)
    return (
        recorded == identity
        and entry.get("content_sha256", identity.content_sha256)
        == identity.content_sha256
    )


class IdentityReason(StrEnum):
    UNBOUND = "sweep.accepted_content_unknown"


class AcceptedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_key: str
    recipe: Recipe
    library_commit: str | None = None

    @classmethod
    def read(cls, record: Mapping[str, Any] | None) -> AcceptedRequest | None:
        # Older saved records cannot be bound to whatever the catalog names now.
        if not record or record.get("accepted_content") is None:
            return None
        return cls.model_validate(record["accepted_content"])
