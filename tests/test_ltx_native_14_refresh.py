from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

from generated_catalog import GENERATED

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "ltx-2-19b-dev-fp4-pytorch-single": "adapters/video/ltx2-pytorch",
    "ltx-2-19b-dev-bf16-diffusers-single": "adapters/video/ltx23-sync-native-disk",
    "ltx-2-19b-distilled-diffusers-single": "adapters/video/ltx2-sync-native",
    "ltx-2-19b-distilled-fp8-diffusers-single": "adapters/video/ltx2-sync-native",
    "ltx-2-3-22b-distilled-1-1-diffusers-single": "adapters/video/ltx23-sync-native-disk",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class LtxNative14RefreshTests(unittest.TestCase):
    def test_all_native_recipes_bind_source_bundle_and_package(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for slug, adapter in RECIPES.items():
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            context = recipe["execution"]["build"]["context"]
            self.assertEqual(context["path"], adapter)
            _, _, bundle_digest = tool["source_bundle"](ROOT / adapter)
            self.assertTrue(bundle_digest)
            index = load(GENERATED / "catalog-index.json")
            entry = next(
                item
                for item in index["recipes"]
                if item["source_path"] == f"recipes/{slug}.json"
            )
            self.assertEqual(len(entry["package"]["recipe_content_sha256"]), 64)

    def test_historical_recipes_keep_explicit_candidate_metadata(self) -> None:
        for slug in RECIPES:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertTrue(
                {"executable", "candidate"} <= set(recipe["metadata"]["tags"])
            )
            self.assertTrue(recipe["models"])
            self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_runtime_sources_are_pinned_and_pipeline_runners_verify_shape(self) -> None:
        for slug, adapter in RECIPES.items():
            root = ROOT / adapter
            dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
            runner = (
                root
                / ("pipelines/run.py" if adapter.endswith("ltx2-pytorch") else "run.py")
            ).read_text(encoding="utf-8")
            self.assertIn("2d6e71c88be37b55a2dd698c2dff447edfbe5898", dockerfile)
            self.assertIn(
                "25466ea6d7aad0500fb0bb3abd3f7480cac70e79013582e14b0bb86195b31651",
                dockerfile,
            )
            self.assertIn("sha256sum --check --strict", dockerfile)
            self.assertIn("e6de69b20c098d4df2eb9e0294cede8acc552a3c", dockerfile)
            self.assertIn(
                "e80a989df7d99ba0e857738ee1daea901865b9fb2f77aca5d481a287e45b7195",
                dockerfile,
            )
            self.assertIn(
                "colour-science==0.4.7", (root / "requirements.lock").read_text()
            )
            self.assertIn("_verify_ltx_runtime_contract()", runner, slug)
            self.assertIn('LTX_PIPELINES_VERSION = "1.4.1"', runner, slug)
            self.assertIn('"tiling_config",\n)', runner, slug)
            self.assertNotIn('"video_latent"', runner, slug)


if __name__ == "__main__":
    unittest.main()
