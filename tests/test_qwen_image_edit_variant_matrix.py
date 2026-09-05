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


class QwenImageEditVariantMatrixTests(unittest.TestCase):
    def test_every_edit_variant_resolves_exact_model_and_job_interface(self) -> None:
        paths = sorted((ROOT / "recipes").glob("qwen-image-edit-2511-*.json"))
        self.assertGreaterEqual(len(paths), 5)
        for path in paths:
            with self.subTest(recipe=path.name):
                recipe = read(path)
                self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
                self.assertEqual(recipe["topology"]["node_count"], 1)
                for selection in recipe["models"]:
                    model = read(ROOT / "models" / f"{selection['model']['slug']}.json")  # type: ignore[index]
                    canonical = content_sha256(ModelDefinition.model_validate(model))
                    self.assertEqual(selection["model"]["content_sha256"], canonical)  # type: ignore[index]
                    self.assertTrue(selection["files"])

    def test_edit_variants_have_bounded_offline_resources_and_output_checks(self) -> None:
        for path in sorted((ROOT / "recipes").glob("qwen-image-edit-2511-*.json")):
            recipe = read(path)
            with self.subTest(recipe=path.name):
                self.assertIn(recipe["execution"]["build"]["network"]["mode"], {"none", "public"})  # type: ignore[index]
                memory = recipe["topology"]["roles"][0]["resources"]["memory"]  # type: ignore[index]
                self.assertLessEqual(memory["startup_peak_bytes"] + memory["system_reserve_bytes"], 128_000_000_000)
                checks = recipe["validation"]["serving"]["checks"]  # type: ignore[index]
                self.assertTrue(any("artifact.output" in check["assertions"] for check in checks))


if __name__ == "__main__":
    unittest.main()
