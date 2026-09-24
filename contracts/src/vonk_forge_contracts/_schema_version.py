"""Reusable strict schema-version field annotations."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator


def _strict_version(expected: int) -> BeforeValidator:
    """Accept a schema tag only when it is the exact Python integer expected."""

    def validate(value: object) -> int:
        if type(value) is not int or value != expected:
            raise ValueError(f"schema_version must be the integer {expected}")
        return value

    return BeforeValidator(validate)


# ``ModelDefinition`` and ``RecipeDefinition`` are schema 2; the recipe
# execution ``TestReport`` and recovery coverage receipts are schema 1. The
# qualification authority owns schema 4; the small campaign manifest remains
# schema 2. Each document owns its own version.
SchemaVersion = Annotated[Literal[2], _strict_version(2)]
TestReportSchemaVersion = Annotated[Literal[1], _strict_version(1)]
QualificationAuthoritySchemaVersion = Annotated[Literal[4], _strict_version(4)]
QualificationCampaignSchemaVersion = Annotated[Literal[2], _strict_version(2)]
RecoveryCoverageReceiptSchemaVersion = Annotated[Literal[1], _strict_version(1)]
