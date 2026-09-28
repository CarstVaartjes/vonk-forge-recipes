"""The authored Model documents, keyed the way the contract resolvers take them."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from vonk_forge_contracts import ModelDefinition, document_sha256

ROOT = Path(__file__).resolve().parents[1]


def model_digest(slug: str) -> str:
    """Return the document digest recipes use to reference ``models/<slug>.json``."""

    path = ROOT / "models" / f"{slug}.json"
    return document_sha256(json.loads(path.read_text(encoding="utf-8")))


@cache
def catalog_models() -> dict[str, ModelDefinition]:
    """Every authored Model, keyed by the digest of its published document."""

    models: dict[str, ModelDefinition] = {}
    for path in sorted((ROOT / "models").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        models[document_sha256(document)] = ModelDefinition.model_validate(document)
    return models
