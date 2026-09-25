"""Standalone public contracts for the Vonk Forge recipe library.

The package exports the two author-facing roots, ``ModelDefinition`` and
``RecipeDefinition``, plus the ``TestReport`` that records the execution
evidence for one recipe revision.  The small nested classes are implementation
details of those roots and can evolve without creating a second public
authority.
"""

from __future__ import annotations

from .canonical import content_sha256
from .model import ModelDefinition
from .qualification_authority import (
    QualificationAuthority,
    QualificationBatch,
    QualificationBatchAssignment,
    QualificationCampaignManifest,
    RecipeAuthorityRow,
    RecoveryCoverage,
    RecoveryCoverageConsumption,
    RecoveryCoverageDefinition,
    RecoveryCoverageMember,
    RecoveryCoverageReceipt,
    RecoveryCoverageReceiptEnvelope,
    RecoveryCoverageRef,
    RecoveryCoverageUse,
    RecoveryNodeBuildIdentity,
    RecoveryNodeEvidence,
    RecoveryRankEvidence,
    campaign_authority_json_schema,
    campaign_manifest_json_schema,
    recovery_coverage_consumption_json_schema,
    recovery_coverage_id,
    recovery_coverage_receipt_json_schema,
    recovery_coverage_use_json_schema,
    recovery_receipt_sha256,
)
from .recipe import RecipeDefinition
from .recipe_test_report import TestReport

__version__ = "0.1.0"
CONTRACT_VERSION = 2

__all__ = [
    "ModelDefinition",
    "QualificationAuthority",
    "QualificationBatch",
    "QualificationBatchAssignment",
    "QualificationCampaignManifest",
    "RecipeAuthorityRow",
    "RecipeDefinition",
    "RecoveryCoverage",
    "RecoveryCoverageConsumption",
    "RecoveryCoverageDefinition",
    "RecoveryCoverageMember",
    "RecoveryCoverageReceipt",
    "RecoveryCoverageReceiptEnvelope",
    "RecoveryCoverageRef",
    "RecoveryCoverageUse",
    "RecoveryNodeBuildIdentity",
    "RecoveryNodeEvidence",
    "RecoveryRankEvidence",
    "TestReport",
    "campaign_authority_json_schema",
    "campaign_manifest_json_schema",
    "content_sha256",
    "recovery_coverage_consumption_json_schema",
    "recovery_coverage_id",
    "recovery_coverage_receipt_json_schema",
    "recovery_coverage_use_json_schema",
    "recovery_receipt_sha256",
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


def test_report_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`TestReport`.

    The identifier is the published catalog-contract URL rather than the
    ``/contracts/`` namespace used by the authoring roots, because this
    document is served to publishers as ``schemas/test-report/v1``.
    """

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/schemas/test-report/v1.schema.json",
        **TestReport.model_json_schema(ref_template="#/$defs/{model}"),
    }


# This is a schema accessor, not a pytest test, and consumers import it into
# test modules.  Tell pytest not to collect it.
test_report_json_schema.__test__ = False
