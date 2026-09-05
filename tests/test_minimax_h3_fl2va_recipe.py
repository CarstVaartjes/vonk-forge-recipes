from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes/minimax-h3-fl2va-diffusers-single.json"
FULL_RECIPE_PATH = ROOT / "recipes/minimax-h3-diffusers-single.json"
MODEL_PATH = ROOT / "models/minimax-h3-fl2va-42ed227e.json"
ADAPTER_PATH = ROOT / "adapters/video/minimax-h3-fl2va-modular-diffusers/minimax_h3.py"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class MiniMaxH3Fl2vaRecipeTests(unittest.TestCase):
    def test_variant_is_separate_candidate_with_bounded_input_contract(self) -> None:
        recipe, full = load(RECIPE_PATH), load(FULL_RECIPE_PATH)
        self.assertEqual(recipe["identity"]["slug"], "minimax-h3-fl2va-diffusers-single")
        self.assertTrue({"executable", "candidate", "fl2va"} <= set(recipe["metadata"]["tags"]))
        self.assertNotEqual(recipe["identity"]["slug"], full["identity"]["slug"])
        self.assertEqual(recipe["interfaces"][0]["adapter"], "video-job")

    def test_model_authority_and_filtered_selection_are_exact(self) -> None:
        model, recipe = load(MODEL_PATH), load(RECIPE_PATH)
        from vonk_forge_contracts import ModelDefinition
        canonical = ModelDefinition.model_validate(model).model_dump(mode="json")
        digest = __import__("hashlib").sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest)
        self.assertEqual(model["source"]["revision"], "42ed227ee7df40d41602854ae760620d6eb651fe")
        selected = {item["file_id"] for item in recipe["models"][0]["files"]}
        self.assertEqual(selected, {item["id"] for item in model["files"]})

    def test_adapter_rejects_ref2va_request_before_loading(self) -> None:
        module = runpy.run_path(str(ADAPTER_PATH))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module["_load_request"].__globals__["INPUT_ROOT"] = root
            (root / "request.json").write_text('{"references":[{"type":"image","path":"subject.png"}]}')
            with self.assertRaisesRegex(ValueError, "FL2VA-only variant"): module["_load_request"]()

    def test_resource_contract_uses_exact_single_spark_closure(self) -> None:
        recipe = load(RECIPE_PATH); role = recipe["topology"]["roles"][0]
        self.assertEqual(recipe["topology"]["node_count"], 1)
        self.assertGreater(role["resources"]["disk"]["artifact_bytes"], 0)
        self.assertGreater(role["resources"]["memory"]["startup_peak_bytes"], 0)


if __name__ == "__main__": unittest.main()
