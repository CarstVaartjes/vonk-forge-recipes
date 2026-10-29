from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from documents import display_of
from generated_catalog import GENERATED
from vonk_forge_contracts import RecipeDefinition, document_sha256


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def canonical_digest(path: str) -> str:
    return document_sha256(load(path))


def catalog_entry(slug: str) -> dict[str, object]:
    catalog = json.loads((GENERATED / "catalog-index.json").read_text(encoding="utf-8"))
    return next(
        item
        for item in catalog["recipes"]
        if item["document"]["identity"]["slug"] == slug
    )


class RecipeDeploymentGuidanceTests(unittest.TestCase):
    def test_nemotron_profiles_are_unambiguous(self) -> None:
        standard = load("recipes/nemotron-3-5-lightning-30b-a3b-vllm-single.json")
        latency = load(
            "recipes/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single.json"
        )
        nano = load("recipes/nemotron-3-nano-30b-a3b-vllm-single.json")

        self.assertTrue(
            {"preferred", "default", "target-only"} <= set(display_of(standard).tags)
        )
        self.assertNotIn("memory-tight", display_of(standard).tags)

        latency_tags = set(display_of(latency).tags)
        self.assertTrue(
            {"high-context", "memory-tight", "specialized", "low-concurrency"}
            <= latency_tags
        )
        self.assertNotIn("default", latency_tags)
        self.assertEqual(latency["settings"]["context_tokens"]["value"], 1_048_576)

        self.assertEqual(standard["topology"]["node_count"], 1)
        self.assertEqual(latency["topology"]["node_count"], 1)

        nano_tags = set(display_of(nano).tags)
        self.assertTrue({"executable", "historical", "superseded"} <= nano_tags)
        self.assertIn("Nemotron 3.5 Lightning", display_of(nano).description)

    def test_moss_is_labeled_as_replay_not_interactive_service(self) -> None:
        recipe = load("recipes/moss-vl-realtime-11b-pytorch-single.json")
        tags = set(display_of(recipe).tags)
        self.assertTrue(
            {"session-replay", "batch-job", "not-interactive-service"} <= tags
        )
        self.assertIn(
            "does not expose the upstream interactive WebSocket",
            display_of(recipe).description,
        )
        self.assertEqual(recipe["interfaces"][0]["adapter"], "artifact-job")
        outputs = {slot["id"] for slot in recipe["interfaces"][0]["output"]["slots"]}
        self.assertEqual(outputs, {"replay", "transcript"})

    def test_mova_360p_is_default_and_720p_is_tight(self) -> None:
        low = load("recipes/mova-360p-diffusers-single.json")
        high = load("recipes/mova-720p-diffusers-single.json")
        self.assertTrue({"preferred", "default"} <= set(display_of(low).tags))
        self.assertNotIn("memory-tight", display_of(low).tags)
        self.assertTrue(
            {"memory-tight", "high-resolution"} <= set(display_of(high).tags)
        )
        self.assertNotIn("default", display_of(high).tags)
        self.assertIn("360p recipe as the default", display_of(high).description)

    def test_muse_runtime_and_limit_are_truthful_and_bound(self) -> None:
        recipe_path = "recipes/muse-glimmer-30b-bf16-vllm-single.json"
        recipe = load(recipe_path)
        self.assertEqual(recipe["runtime"]["engine"], "vllm")
        self.assertIn(
            "combining reasoning with JSON-schema structured output is not supported",
            display_of(recipe).description,
        )

    def test_release_metadata_and_package_bind_current_recipe_digests(self) -> None:
        slugs = (
            "nemotron-3-5-lightning-30b-a3b-vllm-single",
            "nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single",
            "nemotron-3-nano-30b-a3b-vllm-single",
            "moss-vl-realtime-11b-pytorch-single",
            "mova-360p-diffusers-single",
            "mova-720p-diffusers-single",
            "muse-glimmer-30b-bf16-vllm-single",
        )
        for slug in slugs:
            with self.subTest(recipe=slug):
                recipe_path = f"recipes/{slug}.json"
                recipe = RecipeDefinition.model_validate(load(recipe_path))
                entry = catalog_entry(slug)
                self.assertEqual(
                    RecipeDefinition.model_validate(entry["document"]).release,
                    recipe.release,
                )
                digest = canonical_digest(recipe_path)
                self.assertEqual(
                    entry["content_sha256"],
                    digest,
                )
                package = entry["package"]
                self.assertEqual(package["recipe_content_sha256"], digest)
                package_path = GENERATED / package["path"]
                payload = package_path.read_bytes()
                self.assertEqual(len(payload), package["expected_bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), package["sha256"])


if __name__ == "__main__":
    unittest.main()
