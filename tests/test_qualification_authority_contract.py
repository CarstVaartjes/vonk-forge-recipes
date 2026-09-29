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
    campaign_authority_json_schema,
    campaign_manifest_json_schema,
    recovery_coverage_id,
)
from vonk_forge_contracts.qualification_authority import (
    RecoveryCoverageDefinition,
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
                "model_license_refs": [
                    {
                        "key": f"model-{sequence}/test-model",
                        "content_sha256": model_digest,
                        "spdx": "Apache-2.0",
                        "url": "https://example.test/license",
                        "attribution": ["test"],
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
        "schema_version": 5,
        "authority_id": "test-authority",
        "catalog": {
            "repository": "CarstVaartjes/vonk-forge-recipes",
            "commit": "a" * 40,
            "release_tag": "v2.0.0",
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


def test_authority_schema_and_model_bind_every_batch_and_recovery_reference() -> None:
    """The canonical authority and JSON Schema must preserve every exact assignment."""

    document = _authority_document()
    Draft202012Validator(campaign_authority_json_schema()).validate(document)
    parsed = QualificationAuthority.model_validate(document)
    assert parsed.schema_version == 5
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
    member = copy.deepcopy(invalid["recovery_coverage"][0]["members"][1])
    payload = {
        "failure_mode": "single-host-restart",
        "representative_recipe": "aa/ss",
        "members": [member],
        "shared": False,
        "equivalence_rationale": "This member retains its dedicated recovery check.",
    }
    coverage_id = recovery_coverage_id(payload)
    invalid["recovery_coverage"].append({"coverage_id": coverage_id, **payload})
    invalid["recipes"][1]["recovery_coverage_refs"] = [
        {
            "coverage_id": coverage_id,
            "failure_mode": "single-host-restart",
            "representative_recipe": "aa/ss",
            "role": "dedicated",
        }
    ]
    with pytest.raises(
        ValidationError, match="members and row references must be exact"
    ):
        QualificationAuthority.model_validate(invalid)

    invalid = copy.deepcopy(document)
    invalid["recipes"][0]["recovery_coverage_refs"].append(
        {
            **invalid["recipes"][0]["recovery_coverage_refs"][0],
            "coverage_id": "0" * 64,
        }
    )
    with pytest.raises(
        ValidationError, match="exactly one reference per recovery mode"
    ):
        QualificationAuthority.model_validate(invalid)

    invalid = copy.deepcopy(document)
    invalid["recovery_coverage"][0]["members"][1]["topology_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="identical runtime and topology"):
        RecoveryCoverage.model_validate(invalid["recovery_coverage"][0])


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
