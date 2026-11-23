from __future__ import annotations

import hashlib
import json
import unittest

from documents import recipe_display
from generated_catalog import GENERATED
from repo_root import ROOT

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

        assert (
            {"preferred", "default", "target-only"}
            <= set(recipe_display(standard["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        )
        assert (
            "memory-tight" not in recipe_display(standard["identity"]["slug"]).tags  # pyright: ignore[reportIndexIssue]
        )

        latency_tags = set(recipe_display(latency["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        assert {
            "high-context",
            "memory-tight",
            "specialized",
            "low-concurrency",
        } <= latency_tags
        assert "default" not in latency_tags
        assert latency["settings"]["context_tokens"]["value"] == 1048576  # pyright: ignore[reportIndexIssue]

        assert standard["topology"]["node_count"] == 1  # pyright: ignore[reportIndexIssue]
        assert latency["topology"]["node_count"] == 1  # pyright: ignore[reportIndexIssue]

        nano_tags = set(recipe_display(nano["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        assert {"executable", "historical", "superseded"} <= nano_tags

    def test_moss_is_labeled_as_replay_not_interactive_service(self) -> None:
        recipe = load("recipes/moss-vl-realtime-11b-pytorch-single.json")
        tags = set(recipe_display(recipe["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        assert {"session-replay", "batch-job", "not-interactive-service"} <= tags
        assert recipe["interfaces"][0]["adapter"] == "artifact-job"  # pyright: ignore[reportIndexIssue]
        outputs = {slot["id"] for slot in recipe["interfaces"][0]["output"]["slots"]}  # pyright: ignore[reportIndexIssue]
        assert outputs == {"replay", "transcript"}

    def test_mova_360p_is_default_and_720p_is_tight(self) -> None:
        low = load("recipes/mova-360p-diffusers-single.json")
        high = load("recipes/mova-720p-diffusers-single.json")
        assert (
            {"preferred", "default"}
            <= set(recipe_display(low["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        )
        assert "memory-tight" not in recipe_display(low["identity"]["slug"]).tags  # pyright: ignore[reportIndexIssue]
        assert (
            {"memory-tight", "high-resolution"}
            <= set(recipe_display(high["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        )
        assert "default" not in recipe_display(high["identity"]["slug"]).tags  # pyright: ignore[reportIndexIssue]

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
                assert (
                    RecipeDefinition.model_validate(entry["document"]).release
                    == recipe.release
                )
                digest = canonical_digest(recipe_path)
                assert entry["content_sha256"] == digest
                package = entry["package"]
                assert package["recipe_content_sha256"] == digest  # pyright: ignore[reportIndexIssue]
                package_path = GENERATED / package["path"]  # pyright: ignore[reportIndexIssue]
                payload = package_path.read_bytes()
                assert len(payload) == package["expected_bytes"]  # pyright: ignore[reportIndexIssue]
                assert hashlib.sha256(payload).hexdigest() == package["sha256"]  # pyright: ignore[reportIndexIssue]


if __name__ == "__main__":
    unittest.main()
