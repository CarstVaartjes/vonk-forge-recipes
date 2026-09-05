from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {"glm-5-2-aqlm-vllm-triple": 3, "glm-5-2-quanttrio-vllm-four": 4, "glm-5-3-flash-nvfp4-vllm-four": 4, "inkling-small-nvfp4-sglang-dual": 2}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class DistributedRecipeAvailabilityTests(unittest.TestCase):
    def test_larger_topologies_and_candidate_metadata_are_explicit(self) -> None:
        for slug, nodes in RECIPES.items():
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertEqual(recipe["topology"]["node_count"], nodes)
            self.assertTrue({"candidate", "executable"} <= set(recipe["metadata"]["tags"]))
            self.assertTrue(recipe["models"])

    def test_per_node_disk_envelopes_are_bounded(self) -> None:
        for slug in RECIPES:
            disk = load(ROOT / "recipes" / f"{slug}.json")["topology"]["roles"][0]["resources"]["disk"]
            self.assertGreater(disk["artifact_bytes"], 0)
            self.assertGreater(disk["staging_bytes"], 0)

    def test_catalog_packages_bind_the_exact_current_recipes(self) -> None:
        index = load(ROOT / "catalog-index.json")
        entries = {Path(item["source_path"]).stem: item for item in index["recipes"]}
        for slug in RECIPES:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertEqual(entries[slug]["package"]["recipe_content_sha256"], digest(recipe))


if __name__ == "__main__": unittest.main()
