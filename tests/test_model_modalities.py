"""The Model modality guard: capabilities and input modalities come together."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = runpy.run_path(str(ROOT / "tools/check-model-modalities"))


def test_understanding_capabilities_and_input_modalities_agree() -> None:
    check = GUARD["disagreements"]
    assert (
        check(
            {
                "capabilities": ["chat", "image-understanding"],
                "modalities": ["image", "text"],
            }
        )
        == []
    )
    assert (
        len(
            check(
                {
                    "capabilities": ["chat", "image-understanding"],
                    "modalities": ["text"],
                }
            )
        )
        == 1
    )
    assert len(check({"capabilities": ["chat"], "modalities": ["text", "video"]})) == 1
    # A d3 decision Model has no text-generation capability and is still covered.
    assert (
        len(
            check(
                {
                    "capabilities": ["decision", "video-understanding"],
                    "modalities": ["image", "text"],
                }
            )
        )
        == 2
    )


def test_generators_are_not_checked_and_only_model_files_are_named() -> None:
    assert (
        GUARD["disagreements"](
            {"capabilities": ["video-generation"], "modalities": ["video"]}
        )
        == []
    )
    assert GUARD["changed_slugs"](
        ["models/a.json", "recipes/b.json", "models/x/c.json"]
    ) == ["a"]
