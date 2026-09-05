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


class Flux2NVFP4RecipeTests(unittest.TestCase):
    def test_official_nvfp4_model_and_recipe_are_exactly_bound(self) -> None:
        model = load("models/flux-2-klein-4b-nvfp4-1db2b2f7.json")
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        self.assertEqual(model["source"]["revision"], "1db2b2f776c24b76f1122e5f69ab1949fc620068")
        self.assertEqual(model["format"]["quantization"], "nvfp4")
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertTrue({"candidate", "executable", "nvfp4"} <= set(recipe["metadata"]["tags"]))

    def test_comfy_workflow_and_interface_remain_hash_locked(self) -> None:
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        workflow = ROOT / "adapters/media/comfyui-core/workflows/flux-2-klein-4b.json"
        self.assertEqual(arguments["workflow-sha256"], hashlib.sha256(workflow.read_bytes()).hexdigest())
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_release_and_target_ledger_bind_current_recipe(self) -> None:
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        release = load("recipe-releases/flux-2-klein-4b-nvfp4-comfyui-single.json")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], digest(recipe))
        target = next(item for item in load("model-targets/image.json")["targets"] if recipe["identity"]["slug"] in item.get("recipe_slugs", []))
        self.assertEqual(target["status"], "candidate")


if __name__ == "__main__":
    unittest.main()
