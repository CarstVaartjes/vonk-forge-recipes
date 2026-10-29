from __future__ import annotations

import json
from pathlib import Path

from vonk_forge_contracts import (
    ModelDefinition,
    ModelDisplay,
    RecipeDefinition,
    RecipeDisplay,
)

ROOT = Path(__file__).resolve().parents[1]


def test_representative_contract_examples_validate() -> None:
    examples = sorted(
        (ROOT / "contracts/src/vonk_forge_contracts/examples").glob("*.json")
    )
    assert {path.name for path in examples} >= {
        "model-definition.json",
        "recipe-source-build.json",
        "recipe-job.json",
        "recipe-dual.json",
        "model-display.json",
        "recipe-display-source-build.json",
    }
    contracts = {
        "model": ModelDefinition,
        "recipe": RecipeDefinition,
        "model-display": ModelDisplay,
        "recipe-display": RecipeDisplay,
    }
    for path in examples:
        document = json.loads(path.read_text(encoding="utf-8"))
        contracts[document["kind"]].model_validate(document)
