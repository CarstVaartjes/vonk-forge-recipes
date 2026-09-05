from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_for(recipe: dict[str, object]) -> dict[str, object]:
    slug = recipe["models"][0]["model"]["slug"]  # type: ignore[index]
    return read(ROOT / "models" / f"{slug}.json")


class QwenImageEditFP8MixedRecipeTests(unittest.TestCase):
    path = ROOT / "recipes/qwen-image-edit-2511-fp8mixed-comfyui-single.json"

    def test_exact_fp8mixed_model_and_selected_file(self) -> None:
        recipe = read(self.path)
        model = model_for(recipe)
        self.assertEqual(model["source"]["revision"], "4c7c4ea236326cbae56d403d22a03c6cd86ad9a0")
        self.assertEqual(model["format"]["quantization"], "fp8mixed")
        self.assertEqual(len(recipe["models"]), 1)
        self.assertEqual(len(recipe["models"][0]["files"]), 1)  # type: ignore[index]
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertEqual(recipe["runtime"]["engine"], "comfyui")

    def test_comfy_workflow_is_pinned_and_offline(self) -> None:
        recipe = read(self.path)
        args = {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}  # type: ignore[index]
        self.assertTrue(args["workflow"].endswith("qwen-image-edit-2511-fp8mixed.json"))
        self.assertEqual(len(args["workflow-sha256"]), 64)
        self.assertIn(recipe["execution"]["build"]["network"]["mode"], {"none", "public"})  # type: ignore[index]
        resources = recipe["topology"]["roles"][0]["resources"]  # type: ignore[index]
        self.assertLessEqual(resources["memory"]["startup_peak_bytes"] + resources["memory"]["system_reserve_bytes"], 128_000_000_000)

    def test_shared_comfyui_recipes_have_self_contained_model_selections(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*-comfyui-single.json")):
            with self.subTest(recipe=path.name):
                recipe = read(path)
                self.assertEqual(recipe["runtime"]["engine"], "comfyui")
                for selection in recipe["models"]:
                    model = read(ROOT / "models" / f"{selection['model']['slug']}.json")  # type: ignore[index]
                    canonical = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    self.assertEqual(selection["model"]["content_sha256"], canonical)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
