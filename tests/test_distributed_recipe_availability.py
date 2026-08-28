from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RECIPES = {
    "glm-5-2-aqlm-vllm-triple": {
        "nodes": 3,
        "version": "2.0.3",
        "alternative": "GLM 5.3 Flash NVFP4 dual",
    },
    "glm-5-2-quanttrio-vllm-four": {
        "nodes": 4,
        "version": "2.0.2",
        "alternative": "GLM 5.3 Flash NVFP4 dual",
    },
    "glm-5-3-flash-nvfp4-vllm-four": {
        "nodes": 4,
        "version": "1.0.1",
        "alternative": "dual-Spark Ray recipe",
    },
    "inkling-975b-a41b-nvfp4-sglang-eight": {
        "nodes": 8,
        "version": "1.0.3",
        "alternative": "Inkling Small NVFP4 dual",
    },
}


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class DistributedRecipeAvailabilityTests(unittest.TestCase):
    def test_larger_topologies_are_explicitly_unavailable_on_two_sparks(self) -> None:
        for slug, expected in RECIPES.items():
            with self.subTest(recipe=slug):
                recipe = load(ROOT / f"recipes/{slug}.json")
                self.assertEqual(recipe["topology"]["node_count"], expected["nodes"])
                description = recipe["metadata"]["description"]
                self.assertIn("cannot run on Vonk's current two-Spark fleet", description)
                self.assertIn(expected["alternative"], description)
                self.assertIn("candidate", recipe["metadata"]["tags"])
                self.assertIn("executable", recipe["metadata"]["tags"])

    def test_metadata_only_releases_bind_the_exact_current_recipes(self) -> None:
        for slug, expected in RECIPES.items():
            with self.subTest(recipe=slug):
                recipe = load(ROOT / f"recipes/{slug}.json")
                release = load(ROOT / f"recipe-releases/{slug}.json")
                self.assertEqual(release["version"], expected["version"])
                self.assertEqual(release["released_at"], "2026-08-28")
                self.assertEqual(release["history"][0]["upgrade_effect"], "metadata-only")
                self.assertEqual(
                    release["history"][0]["recipe_content_sha256"],
                    canonical_digest(recipe),
                )

    def test_quanttrio_is_superseded_without_mutating_historical_contract(self) -> None:
        recipe = load(ROOT / "recipes/glm-5-2-quanttrio-vllm-four.json")
        self.assertTrue({"historical", "superseded", "tp4"} <= set(recipe["metadata"]["tags"]))
        self.assertEqual(
            recipe["artifacts"][0]["repository"],
            "QuantTrio/GLM-5.2-Int4-Int8Mix",
        )
        self.assertEqual(
            recipe["artifacts"][0]["revision"],
            "1d3bcfe5ec549ecd000fd80b37f191183842e983",
        )
        self.assertEqual(
            recipe["runtime"]["distribution"]["slug"],
            "glm-5-2-quanttrio-four-spark",
        )
        release = load(ROOT / "recipe-releases/glm-5-2-quanttrio-vllm-four.json")
        references = release["history"][0]["changes"][0]["references"]
        self.assertTrue(any("keys-latest-GLM-5.2" in item for item in references))

    def test_target_ledger_does_not_recommend_unavailable_topologies(self) -> None:
        target_set = load(ROOT / "model-targets/language.json")
        targets = target_set["targets"]
        by_recipe = {
            slug: target
            for target in targets
            for slug in target.get("recipe_slugs", [])
        }
        self.assertIn("unavailable on the current two-Spark fleet", by_recipe["glm-5-2-aqlm-vllm-triple"]["notes"])
        self.assertIn("Superseded historical TP4", by_recipe["glm-5-2-quanttrio-vllm-four"]["notes"])
        self.assertIn("TP4 recipe is unavailable", by_recipe["glm-5-3-flash-nvfp4-vllm-four"]["notes"])


if __name__ == "__main__":
    unittest.main()
