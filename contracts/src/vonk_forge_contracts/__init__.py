"""Standalone public contracts for the Vonk Forge recipe library.

The package exports the two author-facing roots, ``ModelDefinition`` and
``RecipeDefinition``, and the qualification authority the maintainer campaign
reads. ``CONTRACT_VERSION`` is the semantic version of these contracts, and it
is also the recipe library's release version: an optional or additive change is
a minor release, a breaking change a major one. Recipe content changes never
change it; a release records when its recipes were last updated instead.

Consumers read published documents with ``read_model``/``read_recipe``, which
ignore fields a newer minor release added, and identify a document by the
``document_sha256`` of its published JSON.
"""

from __future__ import annotations

from ._version import __version__
from .canonical import (
    canonical_json,
    document_sha256,
    qualification_fingerprint_sha256,
    read_model,
    read_recipe,
)
from .endpoint_alias import ENDPOINT_ALIAS_PATTERN, EndpointAlias
from .model import (
    GitHubLfsSource,
    GitHubReleaseAsset,
    GitHubReleaseSource,
    ModelDefinition,
)
from .qualification_authority import (
    QualificationAuthority,
    QualificationBatch,
    QualificationBatchAssignment,
    QualificationCampaignManifest,
    RecipeAuthorityRow,
    RecoveryCoverage,
    RecoveryCoverageDefinition,
    RecoveryCoverageMember,
    RecoveryCoverageRef,
    campaign_authority_json_schema,
    campaign_manifest_json_schema,
    recovery_coverage_id,
)
from .qualification_status import (
    PublishedQualificationStatusView,
    QualificationBlame,
    QualificationCauseCode,
    QualificationChange,
    QualificationDerivationInput,
    QualificationEvidenceRecord,
    QualificationFixReference,
    QualificationHardware,
    QualificationIdentity,
    QualificationOutcome,
    QualificationPlatform,
    QualificationQualityNote,
    QualificationRuntimeImage,
    QualificationStatus,
    QualificationStatusView,
    QualificationSummary,
    QualificationTopology,
    QualificationUnverifiedReason,
    RecipeQualificationEvidence,
    derive_status,
    fingerprint_decides,
    published_qualification_status_json_schema,
    published_status,
    qualification_evidence_json_schema,
    qualification_status_json_schema,
    same_execution,
)
from .recipe import RecipeDefinition, RecipeOptionError

CONTRACT_VERSION = __version__
CONTRACT_MAJOR = int(CONTRACT_VERSION.split(".", 1)[0])

__all__ = [
    "CONTRACT_MAJOR",
    "CONTRACT_VERSION",
    "ENDPOINT_ALIAS_PATTERN",
    "EndpointAlias",
    "GitHubLfsSource",
    "GitHubReleaseAsset",
    "GitHubReleaseSource",
    "ModelDefinition",
    "PublishedQualificationStatusView",
    "QualificationAuthority",
    "QualificationBatch",
    "QualificationBatchAssignment",
    "QualificationBlame",
    "QualificationCampaignManifest",
    "QualificationCauseCode",
    "QualificationChange",
    "QualificationDerivationInput",
    "QualificationEvidenceRecord",
    "QualificationFixReference",
    "QualificationHardware",
    "QualificationIdentity",
    "QualificationOutcome",
    "QualificationPlatform",
    "QualificationQualityNote",
    "QualificationRuntimeImage",
    "QualificationStatus",
    "QualificationStatusView",
    "QualificationSummary",
    "QualificationTopology",
    "QualificationUnverifiedReason",
    "RecipeAuthorityRow",
    "RecipeDefinition",
    "RecipeOptionError",
    "RecipeQualificationEvidence",
    "RecoveryCoverage",
    "RecoveryCoverageDefinition",
    "RecoveryCoverageMember",
    "RecoveryCoverageRef",
    "campaign_authority_json_schema",
    "campaign_manifest_json_schema",
    "canonical_json",
    "derive_status",
    "document_sha256",
    "fingerprint_decides",
    "published_qualification_status_json_schema",
    "published_status",
    "qualification_evidence_json_schema",
    "qualification_fingerprint_sha256",
    "qualification_status_json_schema",
    "read_model",
    "read_recipe",
    "recovery_coverage_id",
    "same_execution",
]


def model_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`ModelDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/model-definition-v2.schema.json",
        **ModelDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-definition-v2.schema.json",
        **RecipeDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }
