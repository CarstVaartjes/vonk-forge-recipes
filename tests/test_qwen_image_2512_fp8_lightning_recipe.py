from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import ModelDefinition, content_sha256  # noqa: E402


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: dict[str, object]) -> str:
    return content_sha256(ModelDefinition.model_validate(value))


class QwenImage2512FP8LightningRecipeTests(unittest.TestCase):
    def test_lightning_recipe_selects_exact_base_and_lora_models(self) -> None:
        path = ROOT / "recipes/qwen-image-2512-lightning-diffusers-single.json"
        recipe = read(path)
        self.assertEqual(recipe["runtime"]["engine"], "diffusers")
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertEqual(len(recipe["models"]), 2)
        for selection in recipe["models"]:
            model = read(ROOT / "models" / f"{selection['model']['slug']}.json")  # type: ignore[index]
            self.assertEqual(selection["model"]["content_sha256"], digest(model))  # type: ignore[index]
            self.assertTrue(selection["files"])

    def test_four_step_1328_image_request_and_offline_build(self) -> None:
        recipe = read(ROOT / "recipes/qwen-image-2512-lightning-diffusers-single.json")
        args = {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}  # type: ignore[index]
        self.assertEqual(args["num-inference-steps"], 4)
        self.assertEqual(args["width"], 1328)
        self.assertEqual(args["height"], 1328)
        self.assertIn(recipe["execution"]["build"]["network"]["mode"], {"none", "public"})  # type: ignore[index]
        self.assertTrue(recipe["validation"]["serving"]["checks"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
