from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def catalog_digest(recipe: dict) -> str:
    index = load("catalog-index.json")
    entry = next(item for item in index["recipes"] if item["source_path"] == f"recipes/{recipe['identity']['slug']}.json")
    return entry["package"]["recipe_content_sha256"]


class Flux2NVFP4RecipeTests(unittest.TestCase):
    def test_official_nvfp4_model_and_recipe_are_exactly_bound(self) -> None:
        model = load("models/flux-2-klein-4b-nvfp4-1db2b2f7.json")
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        self.assertEqual(model["source"]["revision"], "1db2b2f776c24b76f1122e5f69ab1949fc620068")
        self.assertEqual(model["format"]["quantization"], "nvfp4")
        from vonk_forge_contracts import ModelDefinition
        canonical = ModelDefinition.model_validate(model).model_dump(mode="json")
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(canonical))
        self.assertTrue({"candidate", "executable", "nvfp4"} <= set(recipe["metadata"]["tags"]))

    def test_comfy_workflow_and_interface_remain_hash_locked(self) -> None:
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        workflow = ROOT / "adapters/media/comfyui-core/workflows/flux-2-klein-4b.json"
        self.assertEqual(arguments["workflow-sha256"], hashlib.sha256(workflow.read_bytes()).hexdigest())
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_catalog_package_binds_current_recipe(self) -> None:
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        self.assertEqual(catalog_digest(recipe), digest(recipe))


if __name__ == "__main__":
    unittest.main()
