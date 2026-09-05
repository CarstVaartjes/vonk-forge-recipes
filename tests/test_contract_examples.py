from __future__ import annotations

import json
from pathlib import Path

from vonk_forge_contracts import ModelDefinition, RecipeDefinition


ROOT = Path(__file__).resolve().parents[1]


def test_representative_contract_examples_validate() -> None:
    examples = sorted((ROOT / "contracts/src/vonk_forge_contracts/examples").glob("*.json"))
    assert {path.name for path in examples} >= {
        "model-definition.json",
        "recipe-image.json",
        "recipe-source-build.json",
        "recipe-job.json",
        "recipe-dual.json",
    }
    for path in examples:
        document = json.loads(path.read_text(encoding="utf-8"))
        (ModelDefinition if document["kind"] == "model" else RecipeDefinition).model_validate(document)
