from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from repo_root import ROOT
from vonk_forge_contracts import document_sha256


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: dict[str, object]) -> str:
    return document_sha256(value)


class QwenImage2512FP8LightningRecipeTests(unittest.TestCase):
    def test_lightning_recipe_selects_exact_base_and_lora_models(self) -> None:
        path = ROOT / "recipes/qwen-image-2512-lightning-diffusers-single.json"
        recipe = read(path)
        self.assertEqual(recipe["runtime"]["engine"], "diffusers")  # pyright: ignore[reportIndexIssue]
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")  # pyright: ignore[reportIndexIssue]
        self.assertEqual(len(recipe["models"]), 2)  # pyright: ignore[reportArgumentType]
        for selection in recipe["models"]:  # pyright: ignore[reportGeneralTypeIssues]
            model = read(ROOT / "models" / f"{selection['model']['slug']}.json")
            self.assertEqual(selection["model"]["content_sha256"], digest(model))
            self.assertTrue(selection["files"])

    def test_four_step_1328_image_request_and_offline_build(self) -> None:
        recipe = read(ROOT / "recipes/qwen-image-2512-lightning-diffusers-single.json")
        args = {
            item["name"]: item.get("value")
            for item in recipe["runtime"]["arguments"]  # pyright: ignore[reportIndexIssue]
        }
        self.assertEqual(args["num-inference-steps"], 4)
        self.assertEqual(args["width"], 1328)
        self.assertEqual(args["height"], 1328)
        validation: Any = recipe["validation"]
        self.assertTrue(validation["serving"]["checks"])


if __name__ == "__main__":
    unittest.main()
