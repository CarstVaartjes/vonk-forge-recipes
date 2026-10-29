"""The one place tools and tests get a Recipe, a Model and their display.

Documents are execution data (``recipes/<slug>.json``, ``models/<slug>.json``).
Display is what a person reads about them. On contracts 2.x the display is
derived from the document; contracts 3.0.0 switches only the two display
functions to read ``display/**``. Callers never read display fields off a
document.
"""

from __future__ import annotations

import json
from pathlib import Path

from vonk_forge_contracts import (
    ModelDefinition,
    ModelDisplay,
    RecipeDefinition,
    RecipeDisplay,
    model_display_of,
    recipe_display_of,
)

ROOT = Path(__file__).resolve().parents[1]


def recipe(slug: str, root: Path = ROOT) -> RecipeDefinition:
    path = root / "recipes" / f"{slug}.json"
    return RecipeDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))


def model(slug: str, root: Path = ROOT) -> ModelDefinition:
    path = root / "models" / f"{slug}.json"
    return ModelDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))


def recipe_display(slug: str, root: Path = ROOT) -> RecipeDisplay:
    return recipe_display_of(recipe(slug, root))


def model_display(slug: str, root: Path = ROOT) -> ModelDisplay:
    return model_display_of(model(slug, root))
