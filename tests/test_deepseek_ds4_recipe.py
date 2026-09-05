from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-ds4-single.json"
RELEASE = ROOT / "recipe-releases/deepseek-v4-flash-0731-ds4-single.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    body = json.dumps(load(path), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


class DeepseekDs4RecipeTests(unittest.TestCase):
    def test_cuda_profile_describes_ordered_two_session_fallback(self) -> None:
        recipe = load(RECIPE)
        self.assertIn("two-session concurrency", recipe["metadata"]["title"])
        self.assertEqual(next(a["setting"] for a in recipe["runtime"]["arguments"] if a["name"] == "batch-size"), "concurrency")
        self.assertEqual(recipe["settings"]["kind"], "generation")
        self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_release_binds_the_current_recipe_digest(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        recipe = load(RECIPE)
        release = tool["recipe_release"](RELEASE, publisher=recipe["identity"]["publisher"], slug=recipe["identity"]["slug"], recipe_digest=digest(RECIPE))
        self.assertEqual(release["history"][0]["recipe_content_sha256"], digest(RECIPE))


if __name__ == "__main__":
    unittest.main()
