"""Every reviewed service alias is an endpoint alias the platform accepts."""

from __future__ import annotations

import pytest

from qualification.definitions_loader import load_definitions
from qualification.service_aliases import (
    invalid_service_aliases,
    require_valid_service_aliases,
)


def test_every_reviewed_service_alias_is_a_valid_endpoint_alias() -> None:
    definitions = load_definitions()
    assert definitions["service_recipes"]
    assert invalid_service_aliases(definitions) == {}


@pytest.mark.parametrize(
    "alias", ["Qwen3-Coder-Next-FP8", "nvidia/Qwen3.6-27B-NVFP4", "-lead", "trail-", ""]
)
def test_an_alias_the_contract_refuses_is_rejected_naming_the_recipe(
    alias: str,
) -> None:
    definitions = {"service_recipes": {"p/a": {"alias": alias}, "p/b": {"alias": "ok"}}}
    assert invalid_service_aliases(definitions) == {"p/a": alias}
    with pytest.raises(ValueError, match="p/a"):
        require_valid_service_aliases(definitions)


def test_a_recipe_without_a_reviewed_alias_is_not_rejected() -> None:
    require_valid_service_aliases({"service_recipes": {"p/a": {"smoke_cases": []}}})
