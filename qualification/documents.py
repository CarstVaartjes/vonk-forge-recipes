"""The one place tools and tests get a Recipe, a Model and their display.

Documents are execution data (``recipes/<slug>.json``, ``models/<slug>.json``).
Display is what a person reads about them (``display/recipes/<slug>.json``,
``display/models/<slug>.json``); a display edit never changes a document digest.
Callers never read display fields off a document. A recipe's API facts
(``api/recipes/<slug>.json``) are a third optional sibling, read the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

from vonk_forge_contracts import (
    ModelDefinition,
    ModelDisplay,
    RecipeApi,
    RecipeDefinition,
    RecipeDisplay,
    document_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def recipe(slug: str, root: Path = ROOT) -> RecipeDefinition:
    path = root / "recipes" / f"{slug}.json"
    return RecipeDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))


def recipe_digest(slug: str, root: Path = ROOT) -> str:
    """The digest of the authored document, as qualification evidence records it."""
    path = root / "recipes" / f"{slug}.json"
    return document_sha256(json.loads(path.read_text(encoding="utf-8")))


def model(slug: str, root: Path = ROOT) -> ModelDefinition:
    path = root / "models" / f"{slug}.json"
    return ModelDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))


def recipe_display(slug: str, root: Path = ROOT) -> RecipeDisplay:
    path = root / "display" / "recipes" / f"{slug}.json"
    display = RecipeDisplay.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if display.identity.slug != slug:
        raise ValueError(f"{path.name} describes {display.identity.slug}")
    return display


def model_display(slug: str, root: Path = ROOT) -> ModelDisplay:
    path = root / "display" / "models" / f"{slug}.json"
    display = ModelDisplay.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if display.identity.slug != slug:
        raise ValueError(f"{path.name} describes {display.identity.slug}")
    return display


def has_recipe_api(slug: str, root: Path = ROOT) -> bool:
    return (root / "api" / "recipes" / f"{slug}.json").is_file()


def recipe_api(slug: str, root: Path = ROOT) -> RecipeApi:
    path = root / "api" / "recipes" / f"{slug}.json"
    api = RecipeApi.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if api.identity.slug != slug:
        raise ValueError(f"{path.name} describes {api.identity.slug}")
    return api
