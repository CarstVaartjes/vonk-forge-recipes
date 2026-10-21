"""Persisted identity of accepted work, independent of mutable catalog paging."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from .catalog import Recipe
from .catalog_contract import RuntimeImageIdentity


class VerdictIdentity(BaseModel):
    """Content certified by one verdict; absent image evidence stays unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_sha256: str
    runtime_image: RuntimeImageIdentity | None = None
    model_digests: tuple[str, ...]
    # Older Controllers omit the pin. Invalidate conservatively on any library
    # publication for failed/deferred verdicts; passes require explicit opt-in.
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


def recorded_identity(entry: Mapping[str, Any]) -> VerdictIdentity | None:
    value = entry.get("verdict_identity")
    return VerdictIdentity.model_validate(value) if value is not None else None


def verdict_matches(
    entry: Mapping[str, Any], recipe: Recipe, *, ignore_library_commit: bool = False
) -> bool:
    identity = content_identity(recipe)
    recorded = recorded_identity(entry)
    if ignore_library_commit and recorded is not None:
        recorded = recorded.model_copy(update={"image_library_commit": None})
        identity = identity.model_copy(update={"image_library_commit": None})
    return (
        recorded == identity
        and entry.get("content_sha256", identity.content_sha256)
        == identity.content_sha256
    )


def legacy_image_unknown(entry: Mapping[str, Any], recipe: Recipe) -> bool:
    """Missing image evidence alone does not disprove a content verdict.

    Legacy content hashes also bind the recipe's model references. Compare any
    independently recorded model digests too, rather than masking known changes.
    """
    recorded = recorded_identity(entry)
    current = content_identity(recipe)
    if entry.get("content_sha256") != current.content_sha256:
        return False
    models = entry.get("model_digests")
    if models is not None and tuple(sorted(set(models))) != current.model_digests:
        return False
    if recorded is None:
        return True
    return (
        recorded.runtime_image is None
        and current.runtime_image is not None
        and recorded.content_sha256 == current.content_sha256
        and recorded.model_digests == current.model_digests
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
