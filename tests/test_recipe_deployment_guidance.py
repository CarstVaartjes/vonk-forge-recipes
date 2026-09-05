from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import RecipeDefinition, content_sha256


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def canonical_digest(path: str) -> str:
    return content_sha256(RecipeDefinition.model_validate(load(path)))


def catalog_entry(slug: str) -> dict[str, object]:
    catalog = load("catalog-index.json")
    return next(
        item for item in catalog["recipes"]
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
            {"preferred", "default", "target-only"} <= set(standard["metadata"]["tags"])
        )
        self.assertNotIn("memory-tight", standard["metadata"]["tags"])

        latency_tags = set(latency["metadata"]["tags"])
        self.assertTrue(
            {"high-context", "memory-tight", "specialized", "low-concurrency"}
            <= latency_tags
        )
        self.assertNotIn("default", latency_tags)
        self.assertEqual(latency["settings"]["context_tokens"]["value"], 1_048_576)

        self.assertEqual(standard["execution"]["mode"], "build")
        self.assertEqual(latency["execution"]["mode"], "build")
        self.assertEqual(standard["topology"]["node_count"], 1)
        self.assertEqual(latency["topology"]["node_count"], 1)

        nano_tags = set(nano["metadata"]["tags"])
        self.assertTrue({"executable", "historical", "superseded"} <= nano_tags)
        self.assertIn("Nemotron 3.5 Lightning", nano["metadata"]["description"])

    def test_moss_is_labeled_as_replay_not_interactive_service(self) -> None:
        recipe = load("recipes/moss-vl-realtime-11b-pytorch-single.json")
        tags = set(recipe["metadata"]["tags"])
        self.assertTrue(
            {"session-replay", "batch-job", "not-interactive-service"} <= tags
        )
        self.assertIn(
            "does not expose the upstream interactive WebSocket",
            recipe["metadata"]["description"],
        )
        self.assertEqual(recipe["interfaces"][0]["adapter"], "artifact-job")
        outputs = {slot["id"] for slot in recipe["interfaces"][0]["output"]["slots"]}
        self.assertEqual(outputs, {"replay", "transcript"})

    def test_mova_360p_is_default_and_720p_is_tight(self) -> None:
        low = load("recipes/mova-360p-diffusers-single.json")
        high = load("recipes/mova-720p-diffusers-single.json")
        self.assertTrue({"preferred", "default"} <= set(low["metadata"]["tags"]))
        self.assertNotIn("memory-tight", low["metadata"]["tags"])
        self.assertTrue(
            {"memory-tight", "high-resolution"} <= set(high["metadata"]["tags"])
        )
        self.assertNotIn("default", high["metadata"]["tags"])
        self.assertIn("360p recipe as the default", high["metadata"]["description"])

    def test_muse_runtime_and_limit_are_truthful_and_bound(self) -> None:
        recipe_path = "recipes/muse-glimmer-30b-bf16-vllm-single.json"
        recipe = load(recipe_path)
        self.assertEqual(recipe["runtime"]["engine"], "vllm")
        self.assertIn(
            "combining reasoning with JSON-schema structured output is not supported",
            recipe["metadata"]["description"],
        )
        self.assertEqual(recipe["execution"]["mode"], "build")

    def test_release_metadata_and_package_bind_current_recipe_digests(self) -> None:
        versions = {
            "nemotron-3-5-lightning-30b-a3b-vllm-single": ("1.3.6", "2026-09-03"),
            "nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single": ("1.1.5", "2026-09-03"),
            "nemotron-3-nano-30b-a3b-vllm-single": ("2.0.6", "2026-09-05"),
            "moss-vl-realtime-11b-pytorch-single": (
                "1.1.6",
                "2026-09-05",
                "70b5a72ac7089b4e00ec6cd602532c36769d1577ea8d1b0cbd4bd1c27742537c",
            ),
            "mova-360p-diffusers-single": ("2.0.8", "2026-09-05"),
            "mova-720p-diffusers-single": ("2.0.8", "2026-09-05"),
            "muse-glimmer-30b-bf16-vllm-single": ("1.0.5", "2026-09-05"),
        }
        for slug, expected in versions.items():
            version, released_at, *prior = expected
            with self.subTest(recipe=slug):
                recipe_path = f"recipes/{slug}.json"
                recipe = RecipeDefinition.model_validate(load(recipe_path))
                release = recipe.release
                self.assertEqual(release.version, version)
                self.assertEqual(release.released_at, released_at)
                self.assertEqual(release.history[0].version, version)
                self.assertEqual(release.history[0].released_at, released_at)
                self.assertEqual(
                    release.history[0].prior_recipe_content_sha256,
                    prior[0] if prior else None,
                )
                self.assertIn(release.history[0].upgrade_effect, {"none", "restart", "reprepare", "rebuild"})
                entry = catalog_entry(slug)
                digest = canonical_digest(recipe_path)
                self.assertEqual(
                    entry["content_sha256"],
                    digest,
                )
                package = entry["package"]
                self.assertEqual(package["recipe_content_sha256"], digest)
                package_path = ROOT / package["path"]
                payload = package_path.read_bytes()
                self.assertEqual(len(payload), package["expected_bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), package["sha256"])


if __name__ == "__main__":
    unittest.main()
