from __future__ import annotations

import json
import runpy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from catalog_documents import catalog_models
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
        self.assertEqual(
            model["source"]["revision"], "1355db6a052410cfd62085d94b58866fd0f2c3c5"
        )
        raw_recipe = read(self.path)
        self.assertEqual(raw_recipe["runtime"]["engine"], "vllm")
        args = {
            item["name"]: item.get("value")
            for item in raw_recipe["runtime"]["arguments"]
        }
        self.assertEqual(args["max-num-batched-tokens"], 8192)
        self.assertEqual(args["moe-backend"], "marlin")
        self.assertEqual(raw_recipe["topology"]["node_count"], 1)

    def test_source_bundle_matches_declared_context(self) -> None:
        recipe = read(self.path)
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        build = recipe["execution"]["build"]
        archive, _, digest = tool["source_bundle"](ROOT / build["context"]["path"])
        self.assertEqual(len(archive), len(archive))
        self.assertRegex(digest, r"^[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
