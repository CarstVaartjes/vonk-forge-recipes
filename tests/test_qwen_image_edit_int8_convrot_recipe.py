from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class QwenImageEditINT8ConvRotRecipeTests(unittest.TestCase):
    path = ROOT / "recipes/qwen-image-edit-2511-int8-convrot-comfyui-single.json"

    def test_exact_official_int8_convrot_model_is_bound(self) -> None:
        recipe = read(self.path)
        selection = recipe["models"][0]
        model = read(ROOT / "models" / f"{selection['model']['slug']}.json")  # type: ignore[index]
        self.assertEqual(model["source"]["revision"], "e9e85de74a8f48c1e3e2656617626348675a2f21")
        self.assertEqual(model["format"]["quantization"], "int8_tensorwise_convrot")
        self.assertEqual(selection["model"]["content_sha256"], __import__("hashlib").sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest())  # type: ignore[index]

    def test_workflow_resource_and_job_contract(self) -> None:
        recipe = read(self.path)
        args = {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}  # type: ignore[index]
        self.assertTrue(args["workflow"].endswith("qwen-image-edit-2511-int8-convrot.json"))
        self.assertEqual(len(args["workflow-sha256"]), 64)
        self.assertEqual(recipe["runtime"]["engine"], "comfyui")
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertIn(recipe["execution"]["build"]["network"]["mode"], {"none", "public"})  # type: ignore[index]
        self.assertTrue(recipe["validation"]["serving"]["checks"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
