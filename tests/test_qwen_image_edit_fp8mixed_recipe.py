from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import document_sha256


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_for(recipe: dict[str, Any]) -> dict[str, object]:
    slug = recipe["models"][0]["model"]["slug"]
    return read(ROOT / "models" / f"{slug}.json")


class QwenImageEditFP8MixedRecipeTests(unittest.TestCase):
    path = ROOT / "recipes/qwen-image-edit-2511-fp8mixed-comfyui-single.json"

    def test_exact_fp8mixed_model_and_selected_file(self) -> None:
        recipe = read(self.path)
        model = model_for(recipe)
        self.assertEqual(
            model["source"]["revision"], "f68ace85e60b4a02a323e394253731947657b7d2"
        )
        self.assertEqual(model["format"]["quantization"], "fp8mixed")
        models: Any = recipe["models"]
        self.assertEqual(models[0]["id"], "primary")
        self.assertEqual(len(models[0]["files"]), 1)
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertEqual(recipe["runtime"]["engine"], "comfyui")

    def test_comfy_workflow_is_pinned_and_offline(self) -> None:
        recipe = read(self.path)
        args = {
            item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]
        }
        self.assertTrue(args["workflow"].endswith("qwen-image-edit-2511-fp8mixed.json"))
        self.assertEqual(len(args["workflow-sha256"]), 64)
        topology: Any = recipe["topology"]
        resources = topology["roles"][0]["resources"]
        self.assertLessEqual(
            resources["memory"]["peak_bytes"] + resources["memory"]["reserve_bytes"],
            128_000_000_000,
        )

    def test_shared_comfyui_recipes_have_self_contained_model_selections(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*-comfyui-single.json")):
            with self.subTest(recipe=path.name):
                recipe = read(path)
                self.assertEqual(recipe["runtime"]["engine"], "comfyui")
                for selection in recipe["models"]:
                    model = read(ROOT / "models" / f"{selection['model']['slug']}.json")
                    canonical = document_sha256(model)
                    self.assertEqual(selection["model"]["content_sha256"], canonical)


if __name__ == "__main__":
    unittest.main()
