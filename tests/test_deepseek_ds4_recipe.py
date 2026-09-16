"""Recipe-structure checks for the DS4 recipes.

The upstream engine's own command line is deliberately not exercised here. The
pinned ds4 source has its own release cadence, and a flag it renames or drops
surfaces when the container runs, not in this producer suite. A check that
compiled the vendored archive and asked its parser about our argument names
would fail on upstream drift while proving nothing about the recipe document.
These tests pin what this repository authors: the declared topology, argument
names and release metadata.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-ds4-single.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    body = json.dumps(
        load(path), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(body.encode()).hexdigest()


class DeepseekDs4RecipeTests(unittest.TestCase):
    def test_cuda_profile_describes_ordered_two_session_fallback(self) -> None:
        recipe = load(RECIPE)
        self.assertIn("two-session concurrency", recipe["metadata"]["title"])
        self.assertEqual(
            next(
                a["setting"]
                for a in recipe["runtime"]["arguments"]
                if a["name"] == "batched-session"
            ),
            "concurrency",
        )
        self.assertEqual(recipe["settings"]["kind"], "generation")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        names = [argument["name"] for argument in recipe["runtime"]["arguments"]]
        self.assertEqual(names, ["model", "ctx", "batched-session"])

    def test_dspark_declares_target_drafter_and_context_arguments(self) -> None:
        recipe = load(
            ROOT / "recipes/deepseek-v4-flash-0731-ds4-dspark-latency-single.json"
        )
        names = [argument["name"] for argument in recipe["runtime"]["arguments"]]
        self.assertEqual(names, ["model", "mtp-model", "ctx"])
        self.assertEqual(recipe["release"]["version"], "1.2.5")

    def test_release_binds_the_current_recipe_digest(self) -> None:
        index = load(ROOT / "catalog-index.json")
        entry = next(
            item
            for item in index["recipes"]
            if item["source_path"] == f"recipes/{RECIPE.name}"
        )
        self.assertEqual(entry["package"]["recipe_content_sha256"], digest(RECIPE))


if __name__ == "__main__":
    unittest.main()
