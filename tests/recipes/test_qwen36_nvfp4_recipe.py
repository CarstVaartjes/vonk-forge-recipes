from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from catalog_documents import catalog_models
from repo_root import ROOT

from recipe_tooling import catalog_index
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.resolver import validate_recipe_models


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class Qwen36Nvfp4RecipeTests(unittest.TestCase):
    path = ROOT / "recipes/qwen3-6-35b-a3b-nvfp4-vllm-single.json"

    def test_exact_model_and_vllm_runtime_profile(self) -> None:
        recipe = RecipeDefinition.model_validate(read(self.path))
        validate_recipe_models(recipe, catalog_models())
        model = read(ROOT / "models/qwen3-6-35b-a3b-nvfp4-1355db6a.json")
        assert (
            model["source"]["revision"]  # pyright: ignore[reportIndexIssue]
            == "1355db6a052410cfd62085d94b58866fd0f2c3c5"
        )
        raw_recipe = read(self.path)
        assert raw_recipe["runtime"]["engine"] == "vllm"  # pyright: ignore[reportIndexIssue]
        args = {
            item["name"]: item.get("value")
            for item in raw_recipe["runtime"]["arguments"]  # pyright: ignore[reportIndexIssue]
        }
        assert args["max-num-batched-tokens"] == 8192
        assert args["moe-backend"] == "marlin"
        assert raw_recipe["topology"]["node_count"] == 1  # pyright: ignore[reportIndexIssue]

    def test_source_bundle_matches_declared_context(self) -> None:
        recipe = read(self.path)
        tool = vars(catalog_index)
        build = recipe["execution"]["build"]  # pyright: ignore[reportIndexIssue]
        _, _, digest = tool["source_bundle"](ROOT / build["context"]["path"])
        assert re.search(r"^[a-f0-9]{64}$", digest)


if __name__ == "__main__":
    unittest.main()
