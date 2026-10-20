"""Canonical, parameter-free media types shared by job producers and validators."""

from collections.abc import Iterable
from typing import Annotated

from pydantic import AfterValidator, Field, StrictStr

MEDIA_TYPE_PATTERN = (
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/"
    r"[a-z0-9][a-z0-9!#$&^_.+-]{0,63}$"
)


def canonical_media_types(values: Iterable[str]) -> list[str]:
    """Sort and deduplicate media types without changing their spelling.

    Case, parameters and syntax are validated at the contract boundary; this
    helper only canonicalizes the ordering and multiplicity of a producer's list.
    """
    return sorted(set(values))


def _require_canonical(values: list[str]) -> list[str]:
    if values != canonical_media_types(values):
        raise ValueError("media types are not canonical (sorted and unique required)")
    return values


MediaType = Annotated[StrictStr, Field(pattern=MEDIA_TYPE_PATTERN)]
CanonicalMediaTypes = Annotated[
    list[MediaType],
    Field(min_length=1, max_length=16),
    AfterValidator(_require_canonical),
]
