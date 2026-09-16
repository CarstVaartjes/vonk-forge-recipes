"""Contract tests for the shared recipe execution test report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))

from vonk_forge_contracts import TestReport, test_report_json_schema

SCHEMA_PATH = (
    ROOT / "contracts/src/vonk_forge_contracts/schema/test-report-v1.schema.json"
)


def report(**overrides: object) -> dict[str, object]:
    """Return one structurally valid report, with overrides applied."""

    document: dict[str, object] = {
        "schema_version": 1,
        "recipe_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "build_input_sha256": "c" * 64,
        "image_digest": "sha256:" + "d" * 64,
        "topology_name": "solo",
        "node_count": 1,
        "runtime": {
            "agent_version": "2.2.0",
            "container_runtime": "docker",
            "architecture": "linux/arm64",
        },
        "checks": [{"name": "spark-job", "passed": True}],
        "started_at": "2026-09-16T10:00:00+00:00",
        "finished_at": "2026-09-16T10:05:30+00:00",
    }
    document.update(overrides)
    return document


def test_valid_report_validates_and_reports_its_overall_verdict() -> None:
    parsed = TestReport.model_validate(report())
    assert parsed.passed is True
    assert parsed.topology_name == "solo"

    failed = TestReport.model_validate(
        report(checks=[{"name": "spark-job", "passed": False}])
    )
    assert failed.passed is False


@pytest.mark.parametrize("version", [True, 1.0, 2, "1", None])
def test_schema_version_must_be_the_exact_integer_one(version: object) -> None:
    with pytest.raises(ValidationError):
        TestReport.model_validate(report(schema_version=version))


def test_schema_version_is_required_evidence() -> None:
    document = report()
    del document["schema_version"]
    with pytest.raises(ValidationError):
        TestReport.model_validate(document)


def test_report_forbids_unknown_fields_and_requires_every_binding() -> None:
    with pytest.raises(ValidationError):
        TestReport.model_validate(report(unexpected="value"))

    for field in (
        "recipe_sha256",
        "source_bundle_sha256",
        "build_input_sha256",
        "image_digest",
        "topology_name",
        "node_count",
        "runtime",
        "checks",
        "started_at",
        "finished_at",
    ):
        document = report()
        del document[field]
        with pytest.raises(ValidationError):
            TestReport.model_validate(document)


@pytest.mark.parametrize("name", ["solo", "dual-spark", "a"])
def test_topology_name_accepts_the_slug_form(name: str) -> None:
    assert TestReport.model_validate(report(topology_name=name)).topology_name == name


@pytest.mark.parametrize("name", ["Solo", "1solo", "-solo", "solo spark", "a" * 65])
def test_topology_name_rejects_anything_else(name: str) -> None:
    with pytest.raises(ValidationError):
        TestReport.model_validate(report(topology_name=name))


def test_checks_must_be_non_empty_and_uniquely_named() -> None:
    with pytest.raises(ValidationError):
        TestReport.model_validate(report(checks=[]))

    with pytest.raises(ValidationError):
        TestReport.model_validate(
            report(
                checks=[
                    {"name": "spark-job", "passed": True},
                    {"name": "spark-job", "passed": False},
                ]
            )
        )

    with pytest.raises(ValidationError):
        TestReport.model_validate(
            report(
                checks=[
                    {"name": f"check-{index}", "passed": True} for index in range(257)
                ]
            )
        )


def test_timestamps_must_be_real_rfc3339_and_in_order() -> None:
    with pytest.raises(ValidationError):
        TestReport.model_validate(
            report(
                started_at="2026-09-16T10:05:30+00:00",
                finished_at="2026-09-16T10:00:00+00:00",
            )
        )

    for started in (
        "2026-09-16 10:00:00+00:00",  # space instead of T
        "2026-09-16T10:00:00",  # missing offset
        "2026-09-16T10:00:00+0000",  # offset without a colon
        "yesterday",
    ):
        with pytest.raises(ValidationError):
            TestReport.model_validate(report(started_at=started))


def test_timestamp_spelling_is_preserved_exactly() -> None:
    spelling = "2026-09-16T10:00:00.500+02:00"
    parsed = TestReport.model_validate(report(started_at=spelling))
    assert parsed.model_dump(mode="json")["started_at"] == spelling


def test_image_digest_requires_the_algorithm_prefix() -> None:
    for digest in ("d" * 64, "sha256:" + "D" * 64, "sha512:" + "d" * 64):
        with pytest.raises(ValidationError):
            TestReport.model_validate(report(image_digest=digest))


def test_node_count_and_runtime_are_strict() -> None:
    for node_count in (0, -1, True):
        with pytest.raises(ValidationError):
            TestReport.model_validate(report(node_count=node_count))

    for runtime in (
        {
            "agent_version": "2.2.0",
            "container_runtime": "containerd",
            "architecture": "linux/arm64",
        },
        {
            "agent_version": "2.2.0",
            "container_runtime": "docker",
            "architecture": "linux/amd64",
        },
        {
            "agent_version": "",
            "container_runtime": "docker",
            "architecture": "linux/arm64",
        },
    ):
        with pytest.raises(ValidationError):
            TestReport.model_validate(report(runtime=runtime))


def test_generated_schema_is_current_and_structural_only() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema == test_report_json_schema()
    assert (
        schema["$id"] == "https://api.vonkforge.ai/schemas/test-report/v1.schema.json"
    )
    assert schema["properties"]["schema_version"] == {
        "const": 1,
        "title": "Schema Version",
        "type": "integer",
    }
    assert "schema_version" in schema["required"]

    validator = Draft202012Validator(schema)
    validator.validate(report())

    # JSON Schema is structural only.  The document below is shaped correctly,
    # so the generated schema accepts it, while the authoritative model refuses
    # a finished-before-started run and a duplicated check name.
    semantically_wrong = report(
        started_at="2026-09-16T10:05:30+00:00",
        finished_at="2026-09-16T10:00:00+00:00",
        checks=[
            {"name": "spark-job", "passed": True},
            {"name": "spark-job", "passed": True},
        ],
    )
    validator.validate(semantically_wrong)
    with pytest.raises(ValidationError):
        TestReport.model_validate(semantically_wrong)
