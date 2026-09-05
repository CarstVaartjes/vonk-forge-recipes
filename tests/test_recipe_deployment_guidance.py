from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def canonical_digest(path: str) -> str:
    payload = json.dumps(
        load(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        arguments = {
            item["name"]: item["value"] for item in latency["runtime"]["arguments"]
        }
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

    def test_metadata_releases_bind_current_recipe_digests(self) -> None:
        versions = {
            "nemotron-3-5-lightning-30b-a3b-vllm-single": ("1.3.5", "2026-09-01", "metadata-only"),
            "nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single": ("1.1.4", "2026-09-01", "metadata-only"),
            "nemotron-3-nano-30b-a3b-vllm-single": ("2.0.4", "2026-09-01", "metadata-only"),
            "moss-vl-realtime-11b-pytorch-single": ("1.1.3", "2026-09-01", "metadata-only"),
            "mova-360p-diffusers-single": ("2.0.6", "2026-09-01", "metadata-only"),
            "mova-720p-diffusers-single": ("2.0.6", "2026-09-01", "metadata-only"),
            "muse-glimmer-30b-bf16-vllm-single": ("1.0.3", "2026-09-01", "metadata-only"),
        }
        for slug, (version, released_at, upgrade_effect) in versions.items():
            with self.subTest(recipe=slug):
                release = load(f"recipe-releases/{slug}.json")
                self.assertEqual(release["version"], release["history"][0]["version"])
                self.assertRegex(release["released_at"], r"^2026-09-")
                self.assertIn(release["history"][0]["upgrade_effect"], {"metadata-only", "restart", "reinstall", "rebuild"})
                self.assertEqual(
                    release["history"][0]["recipe_content_sha256"],
                    canonical_digest(f"recipes/{slug}.json"),
                )


if __name__ == "__main__":
    unittest.main()
