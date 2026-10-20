"""Canonical media lists for recipe producers, independent of Forge's package pin.

The canonicalizer is identical to ``vonk_forge_contracts.media_types`` used by
Forge's ``src/cluster_profiles/cli_artifact_jobs.py`` and
``qualification_fixtures/recipe_cases.py`` (Forge fix/canonical-media-types,
4313837b33e752ed59e33573bdb3c1739115466b). Its source is recipes commit
a0b0ef57d5504bf3ac305aa215cdfa59a2250dc9, contracts/src/vonk_forge_contracts/media_types.py.
Forge's current public-contracts pin (1514c9f4) predates that helper. Keep this
producer local so catalog and sweep fixes work with that pinned package.
"""

import re
from collections.abc import Iterable
from typing import Annotated

from pydantic import AfterValidator, Field, StrictStr, TypeAdapter
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeJobInterface

MEDIA_TYPE_PATTERN = (
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/"
    r"[a-z0-9][a-z0-9!#$&^_.+-]{0,63}$"
)


def canonical_media_types(values: Iterable[str]) -> list[str]:
    """Sort and deduplicate media types without changing their spelling.

    Reject unsupported spellings rather than silently changing an allowed type
    or dropping its parameters. The same syntax is enforced by the wire contract.
    """
    canonical = sorted(set(values))
    if any(re.fullmatch(MEDIA_TYPE_PATTERN, value) is None for value in canonical):
        raise ValueError("media types must be lowercase and parameter-free")
    return canonical


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


_MEDIA_TYPES = TypeAdapter(CanonicalMediaTypes)


def require_canonical_recipe_media(recipe: RecipeDefinition) -> None:
    """Reject noncanonical authored lists before packaging with existing contracts."""
    for interface in recipe.interfaces:
        if not isinstance(interface, RecipeJobInterface):
            continue
        if interface.input is not None:
            _MEDIA_TYPES.validate_python(interface.input.media_types)
            for slot in interface.input.slots or []:
                _MEDIA_TYPES.validate_python(slot.media_types)
        for slot in interface.output.slots:
            _MEDIA_TYPES.validate_python(slot.media_types)
