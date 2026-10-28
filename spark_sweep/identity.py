"""Persisted identity of accepted work, independent of mutable catalog paging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from vonk_forge_contracts import (
    QualificationIdentity,
    QualificationRuntimeImage,
    document_sha256,
    fingerprint_decides,
    same_execution,
)
from vonk_forge_contracts.sweep import SweepWait

from qualification.model_index import ROOT, ModelIndex, model_index

from .catalog import Recipe
from .catalog_contract import RuntimeImageIdentity
from .recovery import Fingerprint, Owner


class VerdictIdentity(BaseModel):
    """Content certified by one verdict; absent image evidence stays unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_sha256: str
    # Recorded with a verdict for the evidence; a verdict still matches by exact
    # content, so verdicts recorded without it keep matching.
    qualification_fingerprint_sha256: str | None = None
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
    # Set once the receipt ended without a successor of its own: the recipe is
    # planned again at ``not_before``, which grows with how long it has kept ending.
    ended_since: float | None = None
    not_before: float | None = None

    @property
    def live(self) -> bool:
        """The receipt still waits; only a live receipt can prove or release a wait."""
        return self.not_before is None


def _fingerprint_identity(recipe: Recipe) -> str | None:
    """The fingerprint of the exact catalog bytes the Controller accepted, if held.

    A recipe file or Model this checkout lacks, or holds with other bytes, is a
    miss: the identity then compares by exact document.
    """
    index = model_index(ROOT)
    memo = (recipe.key, recipe.content_sha256, recipe.model_digests)
    path = ROOT / "recipes" / f"{recipe.slug}.json"
    try:
        stat = path.stat()
        signature: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = None
    # A miss is remembered too, until the recipe file changes: one stat per call.
    if memo in index.accepted and index.accepted[memo][0] == signature:
        return index.accepted[memo][1]
    fingerprint = _derive_fingerprint(recipe, path, index)
    index.accepted[memo] = (signature, fingerprint)
    return fingerprint


def _derive_fingerprint(recipe: Recipe, path: Path, index: ModelIndex) -> str | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document_sha256(document) != recipe.content_sha256:
            return None
        models = {m["model"]["content_sha256"] for m in document["models"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if models != set(recipe.model_digests):
        return None
    return index.fingerprint(document, recipe.content_sha256)


def content_identity(recipe: Recipe) -> VerdictIdentity:
    return VerdictIdentity(
        content_sha256=recipe.content_sha256,
        qualification_fingerprint_sha256=_fingerprint_identity(recipe),
        source_bundle_sha256=recipe.source_bundle_sha256,
        runtime_image=recipe.runtime_image,
        model_digests=tuple(sorted(set(recipe.model_digests))),
        image_library_commit=recipe.library_commit
        if recipe.runtime_image is None
        else None,
    )


def current_identity(recipe: Recipe) -> QualificationIdentity:
    identity = content_identity(recipe)
    return _qualification(identity).model_copy(
        update={"required_sparks": recipe.node_count}
    )


def recorded_identity(entry: Mapping[str, Any]) -> VerdictIdentity | None:
    value = entry.get("verdict_identity")
    return VerdictIdentity.model_validate(value) if value is not None else None


def _qualification(identity: VerdictIdentity) -> QualificationIdentity:
    return QualificationIdentity(
        content_sha256=identity.content_sha256,
        qualification_fingerprint_sha256=identity.qualification_fingerprint_sha256,
        source_bundle_sha256=identity.source_bundle_sha256,
        runtime_image=QualificationRuntimeImage.model_validate(
            identity.runtime_image.model_dump()
        )
        if identity.runtime_image is not None
        else None,
        model_digests=identity.model_digests,
    )


def verdict_matches(entry: Mapping[str, Any], recipe: Recipe) -> bool:
    """A verdict holds for a recipe the status derivation calls the same execution."""
    identity = content_identity(recipe)
    recorded = recorded_identity(entry)
    if recorded is None:
        return False
    if (
        recorded.qualification_fingerprint_sha256 is not None
        and identity.qualification_fingerprint_sha256 is not None
    ):
        before, now = _qualification(recorded), _qualification(identity)
        if fingerprint_decides(before, now) and same_execution(before, now):
            # Same execution behind other display fields: the fingerprint decides.
            return recorded.image_library_commit == identity.image_library_commit
    return (
        recorded.model_copy(
            update={
                "qualification_fingerprint_sha256": identity.qualification_fingerprint_sha256
            }
        )
        == identity
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
