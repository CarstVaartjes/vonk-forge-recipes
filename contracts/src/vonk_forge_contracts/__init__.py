"""Standalone public authoring contracts for the Vonk Forge recipe library.

The package intentionally exports only the two author-facing root types.  The
small nested classes are implementation details of those roots and can evolve
without creating a second public authority.
"""
from __future__ import annotations

from .canonical import content_sha256
from .model import ModelDefinition
from .recipe import RecipeDefinition

__version__ = "0.1.0"
CONTRACT_VERSION = 2

__all__ = ["ModelDefinition", "RecipeDefinition", "content_sha256"]


def model_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`ModelDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/model-definition-v2.schema.json",
        **ModelDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recipe_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`RecipeDefinition`."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recipe-definition-v2.schema.json",
        **RecipeDefinition.model_json_schema(ref_template="#/$defs/{model}"),
    }
