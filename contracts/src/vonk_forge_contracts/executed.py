"""Every closed set the platform executes.

A word outside these sets is refused or skipped, never kept: the platform acts on
each member, so a reader cannot pass an unknown one along. The sets a reader keeps
unknown words of live in :mod:`vonk_forge_contracts.shown`.
``tools/check-closed-sets`` keeps every closed set in one of the two modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

ChangeEffect = Literal["none", "restart", "reprepare", "rebuild"]


ServingKind = Literal[
    "openai.health",
    "openai.chat",
    "openai.vision",
    "openai.tools",
    "openai.completion",
    "openai.embedding",
    "openai.systemone",
    "openai.transcription",
    "openai.speech",
    "image-job.output",
    "audio-job.output",
    "video-job.output",
    "mesh-job.output",
    "artifact-job.output",
]


ServingAssertion = Literal[
    "endpoint.healthy",
    "chat.nonempty",
    "chat.output-cap",
    "tools.called",
    "completion.nonempty",
    "completion.output-cap",
    "embedding.nonempty",
    "decision.probabilities",
    "transcription.nonempty",
    "speech.audio",
    "inference.completed",
    "artifact.output",
]


RecoveryFailureMode = Literal[
    "single-host-restart",
    "dual-rank-loss-recovery",
    "dual-host-restart",
]


class CatalogSection(StrEnum):
    MODELS = "catalog_entities"
    RECIPES = "recipes"
    FAMILIES = "families"


class EntryOutcome(StrEnum):
    KEEP = "keep"
    KEEP_UNRECOGNISED = "keep-unrecognised"
    DROP_BLOCK = "drop-block"
    SKIP = "skip"
    REFUSE = "refuse"


class CatalogBlock(StrEnum):
    """An entry block a consumer can do without; an unreadable one is dropped."""

    DISPLAY = "display"
    API = "api"
    QUALIFICATION = "qualification"
    PREBUILT_IMAGE = "prebuilt_image"
    MEASURED_MEMORY = "measured_memory"


class DigestCheck(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    # No digest or document to compare: the entry is malformed, not tampered.
    UNREADABLE = "unreadable"


class FaultReason(StrEnum):
    MISSING = "missing"
    WRONG_TYPE = "wrong-type"
    INVALID_VALUE = "invalid-value"
    # A word this release does not know where it cannot keep it as an
    # ``UnrecognisedWord`` (a discriminator or an execution field).
    UNKNOWN_WORD = "unknown-word"
    DUPLICATE = "duplicate"


class RefusalReason(StrEnum):
    ENVELOPE_INVALID = "envelope-invalid"
    # Another ``kind``, ``schema_version`` or contract major.
    SCHEMA_UNSUPPORTED = "schema-unsupported"
    DIGEST_MISMATCH = "digest-mismatch"


JobAdapter = Literal["image-job", "audio-job", "video-job", "mesh-job", "artifact-job"]


ServingInterface = Literal[
    "openai", "image-job", "audio-job", "video-job", "mesh-job", "artifact-job"
]


HttpMethod = Literal["GET", "POST"]


CoverageRole = Literal["representative", "shared-member", "dedicated"]


ReviewDisposition = Literal["actionable", "capacity-review"]


BatchMode = Literal["paired-single", "single", "exclusive-dual"]
