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


# Model and Recipe documents carry no schema version: the recipe library
# release they are published in names the contract version. The qualification
# authority and its small campaign manifest keep their own document versions.
QualificationAuthoritySchemaVersion = Annotated[Literal[5], _strict_version(5)]
QualificationCampaignSchemaVersion = Annotated[Literal[2], _strict_version(2)]
