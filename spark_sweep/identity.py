"""Persisted identity of accepted work, independent of mutable catalog paging."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from vonk_forge_contracts import QualificationIdentity, QualificationRuntimeImage
from vonk_forge_contracts.sweep import SweepWait

from .catalog import Recipe
from .catalog_contract import RuntimeImageIdentity
from .recovery import Fingerprint, Owner


class VerdictIdentity(BaseModel):
    """Content certified by one verdict; absent image evidence stays unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_sha256: str
    source_bundle_sha256: str | None = None
    runtime_image: RuntimeImageIdentity | None = None
    model_digests: tuple[str, ...]
    # Unknown image pins bind accepted work conservatively to its library for
    # cache/request matching. Qualification decisions use evidence, not this key.
    # Never substitute the Docker base image.
    image_library_commit: str | None = None


class ControllerWaitHold(BaseModel):
    """An expired dependency observation, not a recipe verdict or a retry counter."""

    model_config = ConfigDict(extra="forbid")

    wait: SweepWait
    identity: QualificationIdentity
    application_id: str
    next_check: float
    platform: dict[Owner, Fingerprint] = Field(default_factory=dict)
    # This sweep's own load request the receipt answers: a successor is followed only
    # within it, never into another recipe's load.
    request_key: str | None = None
    # Followed into the Controller's continuation: its first observed wait of the same
    # reason is the new baseline (a successor has its own operation ids).
    followed: bool = False
    # Set once the receipt ended without a successor of its own: the recipe is
    # planned again at ``not_before``, which grows with how long it has kept ending.
    ended_since: float | None = None
    not_before: float | None = None

    @property
    def live(self) -> bool:
        """The receipt still waits; only a live receipt can prove or release a wait."""
        return self.not_before is None


def content_identity(recipe: Recipe) -> VerdictIdentity:
    return VerdictIdentity(
        content_sha256=recipe.content_sha256,
        source_bundle_sha256=recipe.source_bundle_sha256,
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
        source_bundle_sha256=identity.source_bundle_sha256,
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
