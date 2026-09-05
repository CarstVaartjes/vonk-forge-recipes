from __future__ import annotations

import json
import unittest
from pathlib import Path

from vonk_forge_contracts import ModelDefinition

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(model: dict) -> str:
    return __import__("hashlib").sha256(json.dumps(ModelDefinition.model_validate(model).model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class GlmModelInventoryTests(unittest.TestCase):
    def test_current_inventory_and_recipe_select_the_calibrated_snapshot(self) -> None:
        model = load("models/glm-5-3-flash-nvfp4-caca4e6a.json")
        recipe = load("recipes/glm-5-3-flash-nvfp4-vllm-dual.json")
        self.assertEqual(model["source"]["revision"], "caca4e6a4ebbd66f159d3d2fc256683fd6e27177")
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "ray")
        self.assertEqual(recipe["topology"]["node_count"], 2)
        files = {item["path"]: item for item in model["files"]}
        self.assertIn("model-input-scales.safetensors", files)
        self.assertIn("model.safetensors.index.json", files)

    def test_aqlm_inventory_closes_the_full_pinned_snapshot(self) -> None:
        model = load("models/glm-5-2-nvfp4-aqlm-hybrid-53e0082e.json")
        recipe = load("recipes/glm-5-2-aqlm-vllm-triple.json")
        self.assertTrue(any(item["path"].startswith("traces/") for item in model["files"]))
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "ray")
        self.assertEqual(len({item["id"] for item in model["files"]}), len(model["files"]))

    def test_gated_abliterated_inventory_fails_closed_without_fake_artifact(self) -> None:
        model = load("models/glm-5-3-flash-nvfp4-abliterated-d7f8afa8.json")
        recipe = load("recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json")
        self.assertNotIn("snapshot", {item["path"] for item in model["files"]})
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertIn("inventory-blocked", recipe["metadata"]["tags"])


if __name__ == "__main__":
    unittest.main()
