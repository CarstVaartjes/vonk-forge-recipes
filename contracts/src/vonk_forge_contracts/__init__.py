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

from .canonical import canonical_json, document_sha256, read_model, read_recipe
from .endpoint_alias import ENDPOINT_ALIAS_PATTERN, EndpointAlias
from .model import GitHubReleaseAsset, GitHubReleaseSource, ModelDefinition
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
from .recipe import RecipeDefinition, RecipeOptionError

CONTRACT_VERSION = "2.1.0"
__version__ = CONTRACT_VERSION
CONTRACT_MAJOR = int(CONTRACT_VERSION.split(".", 1)[0])

__all__ = [
    "CONTRACT_MAJOR",
    "CONTRACT_VERSION",
    "ENDPOINT_ALIAS_PATTERN",
    "EndpointAlias",
    "GitHubReleaseAsset",
    "GitHubReleaseSource",
    "ModelDefinition",
    "QualificationAuthority",
    "QualificationBatch",
    "QualificationBatchAssignment",
    "QualificationCampaignManifest",
    "RecipeAuthorityRow",
    "RecipeDefinition",
    "RecipeOptionError",
    "RecoveryCoverage",
    "RecoveryCoverageDefinition",
    "RecoveryCoverageMember",
    "RecoveryCoverageRef",
    "campaign_authority_json_schema",
    "campaign_manifest_json_schema",
    "canonical_json",
    "document_sha256",
    "read_model",
    "read_recipe",
    "recovery_coverage_id",
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
