"""The sweep names its assignments in the way the profile contract accepts.

The reviewed service alias is the upstream's own spelling; the Controller takes
a lowercase identifier, refuses a repeated running alias, and names the refused
field. The fake Controller enforces the same rules.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from sweep_fakes import FakeFleet, FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts import ENDPOINT_ALIAS_PATTERN, EndpointAlias

from spark_sweep.policy import refused_profile_field
from spark_sweep.profile_alias import profile_alias, unique_profile_alias

_CONTRACT = TypeAdapter(EndpointAlias)
_CONTRACT_PATTERN = re.compile(ENDPOINT_ALIAS_PATTERN)


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _added_aliases(fleet: FakeFleet) -> list[str]:
    return [
        call[call.index("--as") + 1]
        for number, call in fleet.calls
        if number == 10 and call[:2] == ("profile", "add")
    ]


@pytest.mark.parametrize(
    "name",
    [
        "Qwen3-Coder-Next-FP8",
        "GLM-5.3-Flash-EXL3",
        "nvidia/Qwen3.6-27B-NVFP4",
        "/models/Step-3.7-Flash-NVFP4",
        "UI_Mate",
        "with space and !punctuation?",
        "-._leading-and-trailing-._",
        "x" * 300,
        "ünïcode-Modèle",
        "///",
        "",
    ],
)
def test_every_derived_alias_is_valid_for_the_contract(name: str) -> None:
    alias = profile_alias(name)
    assert _CONTRACT.validate_python(alias) == alias
    assert _CONTRACT.validate_python(unique_profile_alias(alias, "vonk-forge/x"))


def test_an_already_valid_alias_is_unchanged() -> None:
    assert profile_alias("inkling-small") == "inkling-small"
    assert profile_alias("qwen3.8-27b_v2") == "qwen3.8-27b_v2"


def test_a_suffix_keeps_two_recipes_apart_and_is_stable() -> None:
    one = unique_profile_alias("inkling-small", "vonk-forge/inkling-a")
    two = unique_profile_alias("inkling-small", "vonk-forge/inkling-b")
    assert one != two
    assert one == unique_profile_alias("inkling-small", "vonk-forge/inkling-a")


def test_a_reviewed_alias_with_capitals_does_not_fail_the_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("coder", alias="Qwen3-Coder-Next-FP8")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    assert _entry(sweep, "coder")["status"] == "passed"
    assert _added_aliases(fleet)
    assert all(_CONTRACT_PATTERN.fullmatch(a) for a in _added_aliases(fleet))


def test_variants_that_share_an_alias_are_named_apart_in_one_profile(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("inkling-a", alias="inkling-small"),
        FakeRecipe("inkling-b", alias="inkling-small"),
    ]
    sweep, fleet, _ = make_sweep(tmp_path, recipes, [FakeModel("m1")], gateway=gateway)
    sweep.run()
    assert _entry(sweep, "inkling-a")["status"] == "passed"
    assert _entry(sweep, "inkling-b")["status"] == "passed"
    assert len(set(_added_aliases(fleet))) == 2


def test_a_refusal_of_the_sweeps_own_field_fails_no_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.faults = [
        (
            ("profile", "add"),
            "control.api_error",
            (
                "document does not match the canonical FleetProfileInput contract: "
                "$.assignments[0].assignment_name: violates pattern '^[a-z0-9]'"
            ),
        )
    ]
    seen: list[str] = []
    sweep.clock.hooks.append(lambda _t: seen.extend(sweep.state.data["infra"]))
    assert sweep.run() == 0
    assert _entry(sweep, "a")["status"] == "passed"
    assert _entry(sweep, "a")["attempts"] == 1
    assert "profile-edit" in seen, "named by its real step, not as a review"


def test_a_refusal_of_recipe_data_fails_the_recipe_at_the_profile_edit(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.faults = [
        (
            ("profile", "add"),
            "recipe.invalid_option",
            (
                "request is invalid: body.assignments.0.option_choices: Input "
                "should be a valid dictionary"
            ),
        )
    ]
    sweep.run()
    entry = _entry(sweep, "a")
    assert (entry["status"], entry["phase"], entry["failure_class"]) == (
        "failed",
        "profile-edit",
        "recipe-data",
    )


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("$.assignments[0].assignment_name: violates pattern 'x'", "assignment_name"),
        ("request is invalid: body.assignments.3.spark_ids: too short", "spark_ids"),
        ("$.assignments[1].option_choices.drafter: violates pattern", "option_choices"),
        ("$.name: violates maxLength 120", "name"),
        ("request is invalid", None),
        ("document does not match the canonical FleetProfileInput contract", None),
    ],
)
def test_the_refused_field_is_read_from_the_error(text: str, field: str | None) -> None:
    assert refused_profile_field(text) == field
