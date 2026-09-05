from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("laguna-s-2-1-nvfp4-vllm-single", "laguna-s-2-1-nvfp4-vllm-low-memory-canary-single"), ("nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single", "nemotron-3-5-lightning-dspark-lowmem-canary-single"))


def load(slug: str, directory: str = "recipes") -> dict:
    return json.loads((ROOT / directory / f"{slug}.json").read_text())


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class LowMemoryCanaryRecipeTests(unittest.TestCase):
    def test_original_profiles_are_preserved_and_canaries_are_distinct(self) -> None:
        for original_slug, canary_slug in PAIRS:
            original, canary = load(original_slug), load(canary_slug)
            self.assertNotEqual(original["identity"]["slug"], canary["identity"]["slug"])
            self.assertEqual(original["models"][0]["model"]["slug"], canary["models"][0]["model"]["slug"])
            self.assertIn("canary", canary["metadata"]["tags"])

    def test_canaries_reuse_exact_model_files_and_build_inputs(self) -> None:
        for original_slug, canary_slug in PAIRS:
            original, canary = load(original_slug), load(canary_slug)
            self.assertEqual(canary["models"], original["models"])
            self.assertEqual(canary["execution"], original["execution"])
            self.assertEqual(canary["provenance"], original["provenance"])

    def test_canaries_use_explicit_supported_memory_controls(self) -> None:
        for _, slug in PAIRS:
            recipe = load(slug)
            self.assertEqual(recipe["settings"]["concurrency"]["value"], 1)
            names = {item["name"] for item in recipe["runtime"]["arguments"]}
            self.assertTrue({"kv-cache-dtype", "enforce-eager"} <= names)

    def test_release_history_pins_recipe_evidence(self) -> None:
        for _, slug in PAIRS:
            recipe, release = load(slug), load(slug, "recipe-releases")
            self.assertEqual(release["history"][0]["recipe_content_sha256"], digest(recipe))


if __name__ == "__main__": unittest.main()
