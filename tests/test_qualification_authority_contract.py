from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts/src"))

from vonk_forge_contracts import (
    QualificationAuthority,
    RecoveryCoverage,
    RecoveryCoverageConsumption,
    RecoveryCoverageReceipt,
    RecoveryCoverageReceiptEnvelope,
    campaign_authority_json_schema,
    campaign_manifest_json_schema,
    recovery_coverage_consumption_json_schema,
    recovery_coverage_id,
    recovery_coverage_receipt_json_schema,
    recovery_coverage_use_json_schema,
    recovery_receipt_sha256,
)
from vonk_forge_contracts.qualification_authority import (
    RecoveryCoverageDefinition,
)

INVALIDATED_BY = sorted(
    {
        "recipe_content_sha256",
        "package_sha256",
        "model_content_sha256s",
        "runtime_stack_sha256",
        "topology_sha256",
        "coverage_membership",
        "runtime_image_digest",
        "platform_build_sha256",
        "agent_build_sha256",
        "target_node_ids",
        "smoke_receipt_sha256",
    }
)


def _member(
    recipe: str,
    recipe_digest: str,
    package_digest: str,
    model_digest: str,
    *,
    runtime_stack: str = "c" * 64,
    topology: str = "d" * 64,
) -> dict[str, Any]:
    return {
        "recipe": recipe,
        "recipe_content_sha256": recipe_digest,
        "package_sha256": package_digest,
        "model_content_sha256s": [model_digest],
        "runtime_stack_sha256": runtime_stack,
        "topology_sha256": topology,
    }


def _coverage_payload() -> dict[str, Any]:
    return {
        "failure_mode": "single-host-restart",
        "representative_recipe": "aa/rr",
        "members": [
            _member("aa/rr", "1" * 64, "2" * 64, "3" * 64),
            _member("aa/ss", "4" * 64, "5" * 64, "6" * 64),
        ],
        "shared": True,
        "equivalence_rationale": (
            "Identical runtime and topology identities make host restart recovery "
            "behavior equivalent; each member retains its own package, model and smoke evidence."
        ),
        "invalidated_by": INVALIDATED_BY,
    }


def _authority_document() -> dict[str, Any]:
    payload = _coverage_payload()
    coverage_id = recovery_coverage_id(payload)
    coverage = {"coverage_id": coverage_id, **payload}
    rows = []
    for sequence, member in enumerate(payload["members"], start=1):
        recipe = member["recipe"]
        role = "representative" if recipe == "aa/rr" else "shared-member"
        model_digest = member["model_content_sha256s"][0]
        rows.append(
            {
                "sequence": sequence,
                "key": recipe,
                "content_sha256": member["recipe_content_sha256"],
                "node_count": 1,
                "interface": "openai-service",
                "recipe_version": "1.0.0",
                "package": {
                    "path": f"packages/{recipe.replace('/', '-')}.tar.gz",
                    "sha256": member["package_sha256"],
                    "expected_bytes": 10,
                    "media_type": "application/vnd.vonk-forge.recipe-package.v2+tar+gzip",
                },
                "disposition": "actionable",
                "operator_acceptance_required": False,
                "model_license_refs": [
                    {
                        "key": f"model-{sequence}/test-model",
                        "content_sha256": model_digest,
                        "spdx": "Apache-2.0",
                        "url": "https://example.test/license",
                        "attribution": ["test"],
                        "operator_acceptance_required": False,
                    }
                ],
                "qualification_inputs": ["tiny.png"],
                "smoke_cases": ["default"],
                "review_gates": [],
                "runtime_stack_sha256": member["runtime_stack_sha256"],
                "topology_sha256": member["topology_sha256"],
                "recovery_coverage_refs": [
                    {
                        "coverage_id": coverage_id,
                        "failure_mode": payload["failure_mode"],
                        "representative_recipe": payload["representative_recipe"],
                        "role": role,
                    }
                ],
            }
        )
    return {
        "schema_version": 4,
        "authority_id": "test-authority",
        "catalog": {
            "repository": "CarstVaartjes/vonk-forge-recipes",
            "commit": "a" * 40,
            "release_tag": "v1.0.0",
            "source_commit": "b" * 40,
            "catalog_index_sha256": "8" * 64,
            "qualification_index_sha256": "9" * 64,
            "recipe_count": 2,
        },
        "scope": {
            "maximum_node_count": 2,
            "recipe_count": 2,
            "excluded_topology_recipe_keys": [],
        },
        "recipes": rows,
        "batches": [
            {
                "sequence": 1,
                "id": "batch-001",
                "mode": "paired-single",
                "assignments": [
                    {"recipe": "aa/rr", "lane": 1, "node_count": 1},
                    {"recipe": "aa/ss", "lane": 2, "node_count": 1},
                ],
            }
        ],
        "recovery_coverage": [coverage],
    }


def _recovery_receipt(authority: dict[str, Any]) -> dict[str, Any]:
    coverage = authority["recovery_coverage"][0]
    representative = next(
        member
        for member in coverage["members"]
        if member["recipe"] == coverage["representative_recipe"]
    )
    node = {
        "node_id": "spk_" + "1" * 32,
        "agent_build_sha256": "7" * 64,
        "baseline_boot_id": "boot-before",
        "recovered_boot_id": "boot-after",
        "baseline_event_id": "fleet-baseline-1",
        "offline_event_id": "fleet-offline-1",
        "recovered_event_id": "fleet-recovered-1",
    }
    return {
        "schema_version": 1,
        "coverage_id": coverage["coverage_id"],
        "failure_mode": coverage["failure_mode"],
        "representative_recipe": representative["recipe"],
        "recipe_content_sha256": representative["recipe_content_sha256"],
        "package_sha256": representative["package_sha256"],
        "model_content_sha256s": representative["model_content_sha256s"],
        "runtime_stack_sha256": representative["runtime_stack_sha256"],
        "topology_sha256": representative["topology_sha256"],
        "runtime_image_digest": "sha256:" + "a" * 64,
        "platform_build_sha256": "b" * 64,
        "architecture": "linux/arm64",
        "smoke_receipt_sha256": "c" * 64,
        "checkpoint_event_ids": [
            node["baseline_event_id"],
            node["offline_event_id"],
            node["recovered_event_id"],
        ],
        "nodes": [node],
        "passed": True,
    }


def _consumption_documents() -> dict[str, Any]:
    authority = _authority_document()
    coverage = authority["recovery_coverage"][0]
    receipt = RecoveryCoverageReceipt.model_validate(_recovery_receipt(authority))
    receipt_sha256 = recovery_receipt_sha256(receipt)
    envelope = {
        "receipt": receipt.model_dump(mode="json"),
        "receipt_sha256": receipt_sha256,
    }
    member = next(item for item in coverage["members"] if item["recipe"] == "aa/ss")
    use = {
        "schema_version": 1,
        "coverage_id": coverage["coverage_id"],
        "failure_mode": coverage["failure_mode"],
        "member_recipe": member["recipe"],
        "representative_recipe": coverage["representative_recipe"],
        "representative_receipt_sha256": receipt_sha256,
        "member_recipe_content_sha256": member["recipe_content_sha256"],
        "member_package_sha256": member["package_sha256"],
        "member_model_content_sha256s": member["model_content_sha256s"],
        "member_runtime_stack_sha256": member["runtime_stack_sha256"],
        "member_topology_sha256": member["topology_sha256"],
        "member_runtime_image_digest": receipt.runtime_image_digest,
        "member_platform_build_sha256": receipt.platform_build_sha256,
        "member_nodes": [
            {
                "node_id": receipt.nodes[0].node_id,
                "agent_build_sha256": receipt.nodes[0].agent_build_sha256,
            }
        ],
        "member_smoke_receipt_sha256": "e" * 64,
    }
    return {
        "authority": authority,
        "coverage": coverage,
        "envelope": envelope,
        "use": use,
    }


def test_authority_schema_and_model_bind_every_batch_and_recovery_reference() -> None:
    """The canonical authority and JSON Schema must preserve every exact assignment."""

    document = _authority_document()
    Draft202012Validator(campaign_authority_json_schema()).validate(document)
    parsed = QualificationAuthority.model_validate(document)
    assert parsed.schema_version == 4
    assert parsed.batches[0].mode == "paired-single"
    assert [assignment.recipe for assignment in parsed.batches[0].assignments] == [
        "aa/rr",
        "aa/ss",
    ]
    assert parsed.recovery_coverage[0].shared is True

    invalid = copy.deepcopy(document)
    invalid["batches"][0]["assignments"][1]["node_count"] = 2
    with pytest.raises(ValidationError, match="paired-single"):
        QualificationAuthority.model_validate(invalid)

    invalid = copy.deepcopy(document)
    invalid["recipes"][1]["recovery_coverage_refs"][0]["coverage_id"] = "0" * 64
    with pytest.raises(ValidationError, match="absent recovery definition"):
        QualificationAuthority.model_validate(invalid)

    invalid = copy.deepcopy(document)
    invalid["recovery_coverage"][0]["members"][1]["topology_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="identical runtime and topology"):
        RecoveryCoverage.model_validate(invalid["recovery_coverage"][0])


def test_shared_recovery_consumption_requires_receipt_and_exact_member_identity() -> (
    None
):
    """A valid representative proof cannot credit a changed recipe or Spark build."""

    values = _consumption_documents()
    Draft202012Validator(recovery_coverage_receipt_json_schema()).validate(
        values["envelope"]
    )
    Draft202012Validator(recovery_coverage_use_json_schema()).validate(values["use"])
    consumption = RecoveryCoverageConsumption.model_validate(
        {
            "coverage": values["coverage"],
            "representative_receipt": values["envelope"],
            "use": values["use"],
        }
    )
    Draft202012Validator(recovery_coverage_consumption_json_schema()).validate(
        consumption.model_dump(mode="json")
    )

    invalid = copy.deepcopy(values["use"])
    invalid["member_package_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="member use identity"):
        RecoveryCoverageConsumption.model_validate(
            {
                "coverage": values["coverage"],
                "representative_receipt": values["envelope"],
                "use": invalid,
            }
        )

    invalid = copy.deepcopy(values["use"])
    invalid["member_nodes"][0]["agent_build_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="target Spark or agent build changed"):
        RecoveryCoverageConsumption.model_validate(
            {
                "coverage": values["coverage"],
                "representative_receipt": values["envelope"],
                "use": invalid,
            }
        )

    invalid = copy.deepcopy(values["envelope"])
    invalid["receipt_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="receipt digest is invalid"):
        RecoveryCoverageReceiptEnvelope.model_validate(invalid)


def test_receipt_hash_normalizes_omitted_and_explicit_optional_null() -> None:
    """Declared optional null fields must not create different receipt identities."""

    authority = _authority_document()
    document = _recovery_receipt(authority)
    explicit_null = {**document, "rank_recovery": None}
    omitted_model = RecoveryCoverageReceipt.model_validate(document)
    null_model = RecoveryCoverageReceipt.model_validate(explicit_null)
    assert recovery_receipt_sha256(omitted_model) == recovery_receipt_sha256(null_model)
    assert "rank_recovery" not in omitted_model.model_dump(
        mode="json", exclude_none=True
    )

    envelope = {
        "receipt": explicit_null,
        "receipt_sha256": recovery_receipt_sha256(null_model),
    }
    RecoveryCoverageReceiptEnvelope.model_validate(envelope)

    invalid = _recovery_receipt(authority)
    invalid["checkpoint_event_ids"].append("unbound-event")
    with pytest.raises(ValidationError, match="exactly bind the typed source events"):
        RecoveryCoverageReceipt.model_validate(invalid)

    invalid = _recovery_receipt(authority)
    invalid["nodes"][0]["recovered_boot_id"] = invalid["nodes"][0]["baseline_boot_id"]
    with pytest.raises(ValidationError, match="changed boot ID"):
        RecoveryCoverageReceipt.model_validate(invalid)


def test_dual_rank_recovery_restores_the_lost_rank_without_host_restart_evidence() -> (
    None
):
    """Rank recovery must return on the lost node and bind both participating agents."""

    member = _member("aa/rr", "1" * 64, "2" * 64, "3" * 64)
    definition = {
        "failure_mode": "dual-rank-loss-recovery",
        "representative_recipe": "aa/rr",
        "members": [member],
        "shared": False,
        "equivalence_rationale": "Dual rank loss remains dedicated to this recipe.",
        "invalidated_by": INVALIDATED_BY,
    }
    coverage_id = recovery_coverage_id(definition)
    coverage = {"coverage_id": coverage_id, **definition}
    lost = "spk_" + "1" * 32
    survivor = "spk_" + "2" * 32
    rank_recovery = {
        "lost_node_id": lost,
        "recovered_node_id": lost,
        "survivor_node_id": survivor,
        "node_builds": [
            {"node_id": lost, "agent_build_sha256": "7" * 64},
            {"node_id": survivor, "agent_build_sha256": "8" * 64},
        ],
        "rank_loss_event_id": "rank-loss-event",
        "route_withdrawal_event_id": "route-withdrawal-event",
        "rank_recovery_event_id": "rank-recovery-event",
        "recovered_smoke_receipt_sha256": "9" * 64,
    }
    receipt = {
        "schema_version": 1,
        "coverage_id": coverage_id,
        "failure_mode": "dual-rank-loss-recovery",
        "representative_recipe": "aa/rr",
        "recipe_content_sha256": member["recipe_content_sha256"],
        "package_sha256": member["package_sha256"],
        "model_content_sha256s": member["model_content_sha256s"],
        "runtime_stack_sha256": member["runtime_stack_sha256"],
        "topology_sha256": member["topology_sha256"],
        "runtime_image_digest": "sha256:" + "a" * 64,
        "platform_build_sha256": "b" * 64,
        "architecture": "linux/arm64",
        "smoke_receipt_sha256": "9" * 64,
        "checkpoint_event_ids": [
            "rank-loss-event",
            "route-withdrawal-event",
            "rank-recovery-event",
        ],
        "nodes": [],
        "rank_recovery": rank_recovery,
        "passed": True,
    }
    RecoveryCoverage.model_validate(coverage)
    RecoveryCoverageReceipt.model_validate(receipt)

    invalid = copy.deepcopy(receipt)
    invalid["rank_recovery"]["recovered_node_id"] = survivor
    with pytest.raises(ValidationError, match="restore the lost node"):
        RecoveryCoverageReceipt.model_validate(invalid)

    invalid = copy.deepcopy(receipt)
    invalid["rank_recovery"]["node_builds"].pop()
    with pytest.raises(ValidationError):
        RecoveryCoverageReceipt.model_validate(invalid)


def test_coverage_id_rejects_untyped_or_extra_payload_fields() -> None:
    """Coverage hashes must normalize through the strict typed definition."""

    payload = _coverage_payload()
    assert recovery_coverage_id(payload) == recovery_coverage_id(
        RecoveryCoverageDefinition.model_validate(payload)
    )
    invalid = {**payload, "unreviewed": True}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        recovery_coverage_id(invalid)


def test_campaign_manifest_schema_keeps_optional_defaults_normalized() -> None:
    """The manifest schema and canonical model must agree on declared defaults."""

    manifest = {
        "schema_version": 2,
        "qualification_authority": "../authorities/authority.json",
        "fixture_manifest": "../qualification-index.json",
    }
    Draft202012Validator(campaign_manifest_json_schema()).validate(manifest)
