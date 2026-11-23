"""Reviewed service aliases must be valid endpoint aliases.

The sweep passes ``service_recipes.alias`` as a profile assignment name and the
smoke cases send it as the model name, so an alias the platform contract refuses
fails every load of that recipe. The pattern comes from the published contracts
package, not from a copy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter, ValidationError
from vonk_forge_contracts import EndpointAlias

_ALIAS = TypeAdapter(EndpointAlias)


def invalid_service_aliases(definitions: Mapping[str, Any]) -> dict[str, str]:
    """Recipe id -> alias, for every reviewed alias the contract refuses."""
    invalid: dict[str, str] = {}
    for recipe_id, service in definitions.get("service_recipes", {}).items():
        alias = service.get("alias") if isinstance(service, Mapping) else None
        if alias is None:
            continue
        try:
            _ALIAS.validate_python(alias)
        except ValidationError:
            invalid[recipe_id] = str(alias)
    return invalid


def require_valid_service_aliases(definitions: Mapping[str, Any]) -> None:
    invalid = invalid_service_aliases(definitions)
    if invalid:
        listed = ", ".join(
            f"{key}: {alias!r}" for key, alias in sorted(invalid.items())
        )
        raise ValueError(
            "service aliases must be valid endpoint aliases (lowercase letters, "
            f"digits, '.', '_' and '-'): {listed}"
        )
