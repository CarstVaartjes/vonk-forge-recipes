"""Strict public recipe execution test-report contract.

A test report is the execution evidence for exactly one recipe revision.  It
binds the recipe, source bundle and build inputs by digest to the image that
was actually run, the topology and node count that were exercised, the runtime
that ran it, and the named checks that passed.  The Controller attaches a
report to a catalog revision and requires one before publication export.

This is a publication-evidence contract, not an authoring root: it never
describes model capability and it is not part of ``ModelDefinition`` or
``RecipeDefinition``.  Keeping it here makes one Pydantic definition the single
authority for the document the platform validates and the catalog publishes.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ._schema_version import TestReportSchemaVersion

Sha256 = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
ImageDigest = Annotated[StrictStr, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
TopologyName = Annotated[
    StrictStr,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$"),
]
CheckName = Annotated[StrictStr, Field(max_length=128, pattern=r"^[a-z][a-z0-9_.-]+$")]
# The schema exposes ``format: date-time``; the validator below additionally
# pins the RFC 3339 spelling so a parsed-and-rewritten timestamp cannot change
# the report's bytes.
DateTimeString = Annotated[StrictStr, Field(json_schema_extra={"format": "date-time"})]

_RFC3339 = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class _TestReportContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TestReportRuntime(_TestReportContract):
    """The runtime that executed the run under test."""

    # Contract types, not pytest test cases: consumers import them into test
    # modules, so tell pytest not to collect them.
    __test__ = False

    agent_version: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    container_runtime: Literal["podman", "docker"]
    architecture: Literal["linux/arm64"]


class TestReportCheck(_TestReportContract):
    """One named pass/fail observation from the run."""

    __test__ = False

    name: CheckName
    passed: StrictBool


class TestReport(_TestReportContract):
    """Execution evidence for exactly one recipe revision."""

    __test__ = False

    schema_version: TestReportSchemaVersion
    recipe_sha256: Sha256
    source_bundle_sha256: Sha256
    build_input_sha256: Sha256
    image_digest: ImageDigest
    topology_name: TopologyName
    node_count: Annotated[StrictInt, Field(ge=1)]
    runtime: TestReportRuntime
    checks: list[TestReportCheck] = Field(min_length=1, max_length=256)
    started_at: DateTimeString
    finished_at: DateTimeString

    @field_validator("started_at", "finished_at")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        if _RFC3339.fullmatch(value) is None:
            raise ValueError("timestamps must use RFC 3339 date-time form")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("timestamps must be real date-times") from error
        if parsed.tzinfo is None:
            raise ValueError("timestamps must carry an explicit UTC offset")
        return value

    @model_validator(mode="after")
    def coherent_run(self) -> TestReport:
        if datetime.fromisoformat(self.finished_at) < datetime.fromisoformat(
            self.started_at
        ):
            raise ValueError("finished_at must not precede started_at")
        names = [check.name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("check names must be unique")
        return self

    @property
    def passed(self) -> bool:
        """Whether every named check passed."""

        return all(check.passed for check in self.checks)
