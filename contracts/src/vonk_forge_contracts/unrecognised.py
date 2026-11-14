"""A published word this contract version does not know, kept for display.

A newer minor release may publish a closed word this reader does not list
outside an execution document: a Model's capability or modality, a Recipe's
alignment, an API fact's endpoint, format, tool choice, request placement or
default, or a qualification status, change, reason, cause, quality note,
hardware class or topology. The catalog reader keeps the entry and stores the word as an :class:`UnrecognisedWord`: a ``str`` (so it
displays and serializes as the raw word) that a consumer tells apart with
``isinstance``. Only the consumer readers create one; producer validation
(``model_validate`` without the read context) still refuses an unknown word, and
the published JSON Schema lists only the known words.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticOmit, core_schema

#: The validation context key the consumer readers set (see ``read_context``).
TOLERANT_READ: Final = "vonk_forge_contracts.unrecognised"
#: Set only on the catalog reader's second pass, after it dropped unreadable
#: optional blocks (see ``dropped_context``).
DROPPED_BLOCKS: Final = "vonk_forge_contracts.dropped"


class UnrecognisedWord(str):
    """A word a newer contract minor publishes; the raw text, verbatim."""

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Only an instance, and only inside a consumer reader: a producer can
        # never validate (and so never emit) one, even from a Python dump.
        def validate(value: object, info: ValidationInfo) -> UnrecognisedWord:
            if isinstance(value, cls) and is_reading(info):
                return value
            raise ValueError("an unrecognised word exists only in a consumer read")

        return core_schema.with_info_plain_validator_function(
            validate, serialization=core_schema.to_string_ser_schema()
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # The published schema names only the words this release defines.
        raise PydanticOmit


def read_context() -> dict[str, object]:
    """A fresh validation context under which unknown words are kept."""

    return {TOLERANT_READ: True}


def dropped_context() -> dict[str, object]:
    """The reader's second pass: the dropped optional blocks are ``None``."""

    return {**read_context(), DROPPED_BLOCKS: True}


def is_reading(info: ValidationInfo) -> bool:
    """Whether a consumer reader (``read_context``) runs this validation."""

    context = info.context
    return isinstance(context, dict) and context.get(TOLERANT_READ) is True


def is_dropping(info: ValidationInfo) -> bool:
    """Whether this is the reader's pass after dropping optional blocks."""

    context = info.context
    return (
        isinstance(context, dict)
        and context.get(TOLERANT_READ) is True
        and context.get(DROPPED_BLOCKS) is True
    )


def _keep_unrecognised(
    value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> object:
    try:
        return handler(value)
    except ValueError:
        if is_reading(info) and isinstance(value, str):
            return UnrecognisedWord(value)
        raise


#: Marks a closed-vocabulary field a newer minor may extend, typed
#: ``Annotated[X | UnrecognisedWord, KeepUnrecognised]``: only a consumer
#: reader's context yields the ``UnrecognisedWord``.
KeepUnrecognised: Final = WrapValidator(_keep_unrecognised)
