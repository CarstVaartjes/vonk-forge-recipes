from __future__ import annotations

import json
import hashlib
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


def model(recipe: dict[str, object], index: int = 0) -> dict[str, object]:
    return read(ROOT / "models" / f"{recipe['models'][index]['model']['slug']}.json")  # type: ignore[index]


class OriginAlignedProfileTests(unittest.TestCase):
    def test_profiles_resolve_their_exact_models_and_offline_build_contexts(self) -> None:
        paths = [
            ROOT / "recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json",
            ROOT / "recipes/qwen3-8-flash-next-nvfp4-sglang-dual.json",
            ROOT / "recipes/deepseek-v4-flash-0731-mia-dual.json",
        ]
        models = [ModelDefinition.model_validate(read(p)) for p in sorted((ROOT / "models").glob("*.json"))]
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for path in paths:
            with self.subTest(recipe=path.name):
                recipe = RecipeDefinition.model_validate(read(path))
                validate_recipe_models(recipe, models)
                self.assertEqual(recipe.execution.mode, "build")
                self.assertIn(recipe.execution.build.network.mode, {"none", "public"})
                context = ROOT / recipe.execution.build.context.path
                archive, _, digest = tool["source_bundle"](context)
                self.assertRegex(digest, r"^[a-f0-9]{64}$")
                self.assertEqual(recipe.topology.node_count, 2)

    def test_drowzeys_profile_keeps_1m_runtime_invariants(self) -> None:
        recipe = read(ROOT / "recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json")
        args = {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}  # type: ignore[index]
        self.assertEqual(args["block-size"], 7168)
        self.assertEqual(args["kv-cache-dtype"], "nvfp4_ds_mla")
        self.assertEqual(args["tensor-parallel-size"], 2)
        self.assertEqual(args["max-num-batched-tokens"], 4096)
        self.assertIn("candidate", recipe["metadata"]["tags"])
        self.assertEqual(model(recipe)["license"]["operator_acceptance_required"], True)

    def test_deepseek_profile_keeps_mia_runtime_and_two_model_selection(self) -> None:
        recipe = read(ROOT / "recipes/deepseek-v4-flash-0731-mia-dual.json")
        args = {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}  # type: ignore[index]
        self.assertEqual(args["kv-cache-dtype"], "nvfp4_ds_mla")
        self.assertEqual(args["max-num-batched-tokens"], 8192)
        self.assertEqual(args["moe-backend"], "flashinfer_b12x")
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertIn(model(recipe)["license"]["operator_acceptance_required"], {True, False})


if __name__ == "__main__":
    unittest.main()
