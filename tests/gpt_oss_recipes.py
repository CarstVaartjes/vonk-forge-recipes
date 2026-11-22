"""The recipes that serve a gpt-oss family model, for the gpt-oss guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from documents import model_display
from repo_root import ROOT


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_gpt_oss(recipe: dict[str, Any]) -> bool:
    for binding in recipe.get("models", []):
        lineage = model_display(binding["model"]["slug"]).lineage
        if lineage is not None and lineage.family == "gpt-oss":
            return True
    return False


def gpt_oss_recipes(*engines: str) -> list[Path]:
    """Every gpt-oss recipe whose runtime engine is one of ``engines``."""

    return [
        path
        for path in sorted((ROOT / "recipes").glob("*.json"))
        if read(path).get("runtime", {}).get("engine") in engines
        and is_gpt_oss(read(path))
    ]


def all_arguments(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """The runtime arguments plus every option choice's arguments."""

    arguments = list(recipe["runtime"].get("arguments", []))
    for option in recipe.get("options", []):
        for choice in option.get("choices", []):
            arguments += choice.get("args", [])
    return arguments
