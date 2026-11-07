"""The Model modality guard: capabilities and input modalities come together."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = runpy.run_path(str(ROOT / "tools/check-model-modalities"))


def test_understanding_capabilities_and_input_modalities_agree() -> None:
    check = GUARD["disagreements"]
    assert check(["chat", "image-understanding"], ["image", "text"]) == []
    assert len(check(["chat", "image-understanding"], ["text"])) == 1
    assert len(check(["chat"], ["text", "video"])) == 1
    # A d3 decision Model has no text-generation capability and is still covered.
    assert len(check(["decision", "video-understanding"], ["image", "text"])) == 2


def test_generators_are_not_checked_and_only_model_files_are_named() -> None:
    assert GUARD["disagreements"](["video-generation"], ["video"]) == []
    assert GUARD["changed_slugs"](
        ["display/models/a.json", "models/b.json", "display/models/x/c.json"]
    ) == ["a"]


def test_a_text_generating_model_is_judged_not_skipped() -> None:
    refused = GUARD["disagreements"](
        ["text-generation", "image-understanding"], ["text"]
    )
    assert len(refused) == 1
    code = (["code-generation", "video-understanding"], ["text"])
    assert len(GUARD["disagreements"](*code)) == 1


def test_roles_come_from_the_contract_not_from_name_suffixes() -> None:
    from vonk_forge_contracts import CapabilityName, CapabilityRole

    assert CapabilityName("text-to-speech").role is CapabilityRole.GENERATOR
    assert CapabilityName("image-upscaling").role is CapabilityRole.OTHER
    assert CapabilityName("code-generation").role is CapabilityRole.OTHER
    assert CapabilityName("video-understanding").role is CapabilityRole.UNDERSTANDING
    assert {c for c in CapabilityName if c.role is CapabilityRole.UNDERSTANDING} == set(
        GUARD["INPUT_MODALITY_CAPABILITIES"].values()
    )
