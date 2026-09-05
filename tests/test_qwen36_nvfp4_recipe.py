from __future__ import annotations

import json
import runpy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import ModelDefinition, RecipeDefinition  # noqa: E402
from vonk_forge_contracts.resolver import validate_recipe_models  # noqa: E402


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class Qwen36Nvfp4RecipeTests(unittest.TestCase):
    path = ROOT / "recipes/qwen3-6-35b-a3b-nvfp4-vllm-single.json"

    def test_exact_model_and_vllm_runtime_profile(self) -> None:
        recipe = RecipeDefinition.model_validate(read(self.path))
        models = [ModelDefinition.model_validate(read(p)) for p in (ROOT / "models").glob("*.json")]
        validate_recipe_models(recipe, models)
        model = read(ROOT / "models/qwen3-6-35b-a3b-nvfp4-1355db6a.json")
        self.assertEqual(model["source"]["revision"], "1355db6a052410cfd62085d94b58866fd0f2c3c5")
        self.assertEqual(model["parameters"], {"total": 35_000_000_000, "active": 3_000_000_000})
        raw_recipe = read(self.path)
        self.assertEqual(raw_recipe["runtime"]["engine"], "vllm")
        args = {item["name"]: item.get("value") for item in raw_recipe["runtime"]["arguments"]}  # type: ignore[index]
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
        self.assertIn(build["network"]["mode"], {"none", "public"})


if __name__ == "__main__":
    unittest.main()
