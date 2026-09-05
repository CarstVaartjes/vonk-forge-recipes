from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/catalog-hf-model"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CatalogHfModelSafetyTests(unittest.TestCase):
    def test_catalog_command_exposes_model_authority_options(self) -> None:
        result = __import__("subprocess").run(["/opt/vonk-forge/control/.venv/bin/python", str(TOOL), "--help"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--publisher", "--version-slug", "--architecture", "--quantization"):
            self.assertIn(option, result.stdout)

    def test_current_qwen_snapshot_has_unique_complete_files(self) -> None:
        version = load(ROOT / "models/qwen3-8-27b-fp8-017b9c7a.json")
        files = version["files"]
        self.assertGreater(len(files), 70)
        self.assertEqual(len({item["id"] for item in files}), len(files))
        self.assertEqual(len({item["path"] for item in files}), len(files))

    def test_qwen_dense_models_advertise_native_multimodal_capabilities(self) -> None:
        for slug in ("qwen3-5-9b-c2022362", "qwen3-6-27b-6a9e13bd", "qwen3-8-27b-1d4bf0f2"):
            model = load(ROOT / f"models/{slug}.json")
            capabilities = {fact["capability"] for fact in model["capabilities"]["facts"] if fact["support"] == "supported"}
            self.assertTrue(capabilities & {"text-generation", "image-understanding"}, slug)


if __name__ == "__main__":
    unittest.main()
