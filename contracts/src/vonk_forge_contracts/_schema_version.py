"""Reusable strict schema-version field annotations."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator


def _strict_schema_version(value: object) -> int:
    """Accept the schema tag only when it is the exact Python integer 2."""

    if type(value) is not int or value != 2:
        raise ValueError("schema_version must be the integer 2")
    return value


SchemaVersion = Annotated[Literal[2], BeforeValidator(_strict_schema_version)]
