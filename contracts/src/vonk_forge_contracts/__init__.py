"""Standalone public contracts for the Vonk Forge recipe library.

The package exports the two author-facing roots, ``ModelDefinition`` and
``RecipeDefinition``, and the qualification authority the maintainer campaign
reads. ``CONTRACT_VERSION`` is the semantic version of these contracts, and it
is also the recipe library's release version: an optional or additive change is
a minor release, a breaking change a major one. Recipe content changes never
change it; a release records when its recipes were last updated instead.

Consumers read published documents with ``read_model``/``read_recipe``, which
ignore fields a newer minor release added, and identify a document by the
``document_sha256`` of its published JSON. A published index is read with
``read_catalog``: it keeps every entry this release reads, keeps a word a newer
minor added as an ``UnrecognisedWord``, drops an unreadable optional block (the
entry stays), skips an entry whose execution data is malformed, and refuses only
at integrity edges (``CatalogIndexRefused``).
"""

from __future__ import annotations

from vonk_forge_contracts._version import __version__
from vonk_forge_contracts.api import (
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
from vonk_forge_contracts.api_checks import (
    RecipeApiError,
    check_api_models,
    check_recipe_api,
    endpoint_capabilities,
    served_capabilities,
)
from vonk_forge_contracts.canonical import (
    canonical_json,
    document_sha256,
    read_model,
    read_recipe,
)
from vonk_forge_contracts.capability import (
    CAPABILITY_ROLES,
    CapabilityName,
    CapabilityRole,
)
from vonk_forge_contracts.catalog_index import (
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
from vonk_forge_contracts.catalog_reader import (
    CatalogIndexRefused,
    CatalogRead,
    read_catalog,
    read_catalog_model_entry,
    read_catalog_recipe_entry,
)
from vonk_forge_contracts.display import (
    FamilyDisplay,
    KnobDisplay,
    LineDisplay,
    LineModel,
    ModelDisplay,
    ModelDisplayIdentity,
    ModelLineage,
    RecipeDisplay,
    published_json,
    read_model_display,
    read_recipe_display,
    require_display_of,
)
from vonk_forge_contracts.endpoint_alias import ENDPOINT_ALIAS_PATTERN, EndpointAlias
from vonk_forge_contracts.model import (
    GitHubLfsSource,
    GitHubReleaseAsset,
    GitHubReleaseSource,
    ModelDefinition,
)
from vonk_forge_contracts.qualification_authority import (
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
from vonk_forge_contracts.qualification_status import (
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
from vonk_forge_contracts.recipe import (
    RecipeDefinition,
    RecipeJobKnob,
    RecipeOptionError,
)
from vonk_forge_contracts.unrecognised import UnrecognisedWord
from vonk_forge_contracts.vocabulary import (
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
    "CatalogIndexRefused",
    "CatalogModelEntry",
    "CatalogRead",
    "CatalogRecipeEntry",
    "CatalogVocabulary",
    "ChatApi",
    "DecisionApi",
    "EndpointAlias",
    "FamilyDisplay",
    "GitHubLfsSource",
    "GitHubReleaseAsset",
    "GitHubReleaseSource",
    "KnobDisplay",
    "LineDisplay",
    "LineModel",
    "ModelDefinition",
    "ModelDisplay",
    "ModelDisplayIdentity",
    "ModelLineage",
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
    "RecipeJobKnob",
    "RecipeOptionError",
    "RecipeQualificationEvidence",
    "RecoveryCoverage",
    "RecoveryCoverageDefinition",
    "RecoveryCoverageMember",
    "RecoveryCoverageRef",
    "UnrecognisedWord",
    "campaign_authority_json_schema",
    "campaign_manifest_json_schema",
    "canonical_json",
    "catalog_index_json_schema",
    "catalog_vocabulary",
    "check_api_models",
    "check_recipe_api",
    "derive_status",
    "document_sha256",
    "endpoint_capabilities",
    "published_index_json",
    "published_json",
    "published_measured_memory_json_schema",
    "published_qualification_status_json_schema",
    "published_status",
    "qualification_evidence_json_schema",
    "qualification_status_json_schema",
    "read_catalog",
    "read_catalog_index",
    "read_catalog_model_entry",
    "read_catalog_recipe_entry",
    "read_model",
    "read_model_display",
    "read_recipe",
    "read_recipe_api",
    "read_recipe_display",
    "recovery_coverage_id",
    "require_api_of",
    "require_display_of",
    "served_capabilities",
]


def model_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`ModelDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/model-definition-v5.schema.json",
        **ModelDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-definition-v5.schema.json",
        **RecipeDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_display_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeDisplay`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-display-v5.schema.json",
        **RecipeDisplay.model_json_schema(ref_template="#/$defs/{model}"),
    }


def model_display_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`ModelDisplay`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/model-display-v5.schema.json",
        **ModelDisplay.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_api_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeApi`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-api-v5.schema.json",
        **RecipeApi.model_json_schema(ref_template="#/$defs/{model}"),
    }
