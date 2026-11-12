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
from .api import (
    ApiBlock,
    AudioSpeechApi,
    AudioTranscriptionApi,
    ChatApi,
    DecisionApi,
    RecipeApi,
    RecipeApiOptionChoice,
    read_recipe_api,
    require_api_of,
)
from .api_checks import RecipeApiError, check_api_models, check_recipe_api
from .canonical import canonical_json, document_sha256, read_model, read_recipe
from .capability import CAPABILITY_ROLES, CapabilityName, CapabilityRole
from .catalog_index import (
    INDEX_KIND,
    INDEX_SCHEMA_ASSET,
    INDEX_SCHEMA_VERSION,
    PACKAGE_MEDIA_TYPE,
    PACKAGE_METADATA_FIELDS,
    PACKAGE_PATH_PREFIX,
    PACKAGE_SCHEMA_VERSION,
    CatalogIndex,
    CatalogModelEntry,
    CatalogRecipeEntry,
    PackageContract,
    PackageMetadata,
    PrebuiltImage,
    catalog_index_json_schema,
    published_index_json,
    read_catalog_index,
)
from .display import (
    ModelDisplay,
    ModelDisplayIdentity,
    RecipeDisplay,
    published_json,
    read_model_display,
    read_recipe_display,
    require_display_of,
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
    PublishedMeasuredMemory,
    PublishedQualificationStatusView,
    PublishedRoleMemory,
    QualificationBlame,
    QualificationCauseCode,
    QualificationChange,
    QualificationDerivationInput,
    QualificationEvidenceRecord,
    QualificationFixReference,
    QualificationHardware,
    QualificationIdentity,
    QualificationMeasuredRoleMemory,
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
    published_measured_memory_json_schema,
    published_qualification_status_json_schema,
    published_status,
    qualification_evidence_json_schema,
    qualification_status_json_schema,
)
from .recipe import RecipeDefinition, RecipeOptionError
from .vocabulary import (
    CatalogVocabulary,
    catalog_vocabulary,
)

CONTRACT_VERSION = __version__
CONTRACT_MAJOR = int(CONTRACT_VERSION.split(".", 1)[0])

__all__ = [
    "CAPABILITY_ROLES",
    "CONTRACT_MAJOR",
    "CONTRACT_VERSION",
    "ENDPOINT_ALIAS_PATTERN",
    "INDEX_KIND",
    "INDEX_SCHEMA_ASSET",
    "INDEX_SCHEMA_VERSION",
    "PACKAGE_MEDIA_TYPE",
    "PACKAGE_METADATA_FIELDS",
    "PACKAGE_PATH_PREFIX",
    "PACKAGE_SCHEMA_VERSION",
    "ApiBlock",
    "AudioSpeechApi",
    "AudioTranscriptionApi",
    "CapabilityName",
    "CapabilityRole",
    "CatalogIndex",
    "CatalogModelEntry",
    "CatalogRecipeEntry",
    "CatalogVocabulary",
    "ChatApi",
    "DecisionApi",
    "EndpointAlias",
    "GitHubLfsSource",
    "GitHubReleaseAsset",
    "GitHubReleaseSource",
    "ModelDefinition",
    "ModelDisplay",
    "ModelDisplayIdentity",
    "PackageContract",
    "PackageMetadata",
    "PrebuiltImage",
    "PublishedMeasuredMemory",
    "PublishedQualificationStatusView",
    "PublishedRoleMemory",
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
    "QualificationMeasuredRoleMemory",
    "QualificationOutcome",
    "QualificationPlatform",
    "QualificationQualityNote",
    "QualificationRuntimeImage",
    "QualificationStatus",
    "QualificationStatusView",
    "QualificationSummary",
    "QualificationTopology",
    "QualificationUnverifiedReason",
    "RecipeApi",
    "RecipeApiError",
    "RecipeApiOptionChoice",
    "RecipeAuthorityRow",
    "RecipeDefinition",
    "RecipeDisplay",
    "RecipeOptionError",
    "RecipeQualificationEvidence",
    "RecoveryCoverage",
    "RecoveryCoverageDefinition",
    "RecoveryCoverageMember",
    "RecoveryCoverageRef",
    "campaign_authority_json_schema",
    "campaign_manifest_json_schema",
    "canonical_json",
    "catalog_index_json_schema",
    "catalog_vocabulary",
    "check_api_models",
    "check_recipe_api",
    "derive_status",
    "document_sha256",
    "published_index_json",
    "published_json",
    "published_measured_memory_json_schema",
    "published_qualification_status_json_schema",
    "published_status",
    "qualification_evidence_json_schema",
    "qualification_status_json_schema",
    "read_catalog_index",
    "read_model",
    "read_model_display",
    "read_recipe",
    "read_recipe_api",
    "read_recipe_display",
    "recovery_coverage_id",
    "require_api_of",
    "require_display_of",
]


def model_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`ModelDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/model-definition-v4.schema.json",
        **ModelDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-definition-v4.schema.json",
        **RecipeDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_display_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeDisplay`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-display-v4.schema.json",
        **RecipeDisplay.model_json_schema(ref_template="#/$defs/{model}"),
    }


def model_display_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`ModelDisplay`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/model-display-v4.schema.json",
        **ModelDisplay.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_api_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeApi`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-api-v4.schema.json",
        **RecipeApi.model_json_schema(ref_template="#/$defs/{model}"),
    }
