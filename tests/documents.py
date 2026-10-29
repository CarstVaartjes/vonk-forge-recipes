"""Tests get documents and displays through the repository accessor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vonk_forge_contracts import RecipeDisplay

from qualification.documents import model, model_display, recipe, recipe_display

__all__ = ["display_of", "model", "model_display", "recipe", "recipe_display"]


def display_of(document: Mapping[str, Any]) -> RecipeDisplay:
    """The display of a committed Recipe document a test already holds."""

    return recipe_display(document["identity"]["slug"])
