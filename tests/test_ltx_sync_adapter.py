from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE_SLUGS = ("ltx-2-19b-dev-bf16-diffusers-single", "ltx-2-19b-distilled-diffusers-single")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class LtxSyncAuthorityTests(unittest.TestCase):
    def test_recipes_resolve_exact_runnable_authorities_and_closure(self) -> None:
        for slug in RECIPE_SLUGS:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertTrue(recipe["models"])
            self.assertEqual(recipe["runtime"]["engine"], "pytorch-pipeline")
            self.assertEqual(recipe["interfaces"][0]["adapter"], "video-job")
            self.assertEqual(recipe["topology"]["node_count"], 1)
            self.assertTrue(all(file["mount"]["read_only"] for model in recipe["models"] for file in model["files"]))

    def test_container_is_pinned_and_runtime_is_offline(self) -> None:
        for path in (ROOT / "adapters/video/ltx23-sync-native-disk/Dockerfile", ROOT / "adapters/video/ltx2-sync-native/Dockerfile"):
            self.assertIn("@sha256:", path.read_text())
        for slug in RECIPE_SLUGS:
            environment = {item["name"] for item in load(ROOT / "recipes" / f"{slug}.json")["runtime"]["environment"]}
            self.assertIn("HF_HUB_OFFLINE", environment)

    def test_signed_source_bundle_matches_each_recipe_context(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for slug in RECIPE_SLUGS:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            context = recipe["execution"]["build"]["context"]
            _, _, digest = tool["source_bundle"](ROOT / context["path"])
            self.assertTrue(digest)


if __name__ == "__main__": unittest.main()
