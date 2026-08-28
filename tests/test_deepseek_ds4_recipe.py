from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-ds4-single.json"
RELEASE = ROOT / "recipe-releases/deepseek-v4-flash-0731-ds4-single.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_digest(path: Path) -> str:
    payload = json.dumps(
        load(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DeepseekDs4RecipeTests(unittest.TestCase):
    def test_cuda_profile_describes_ordered_two_session_fallback(self) -> None:
        recipe = load(RECIPE)
        metadata = recipe["metadata"]
        self.assertIn("two-session concurrency", metadata["title"])
        self.assertIn("CUDA's ordered exact fallback", metadata["description"])
        self.assertNotIn("native two-session batching", metadata["description"])
        self.assertEqual(
            next(
                argument["value"]
                for argument in recipe["runtime"]["arguments"]
                if argument["name"] == "batch-size"
            ),
            2,
        )

    def test_release_binds_the_current_recipe_digest(self) -> None:
        recipe = load(RECIPE)
        identity = recipe["identity"]
        release_validator = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "recipe_release"
        ]
        release = release_validator(
            RELEASE,
            publisher=identity["publisher"],
            slug=identity["slug"],
            recipe_digest=canonical_digest(RECIPE),
        )
        self.assertEqual(release["version"], "2.0.2")
        self.assertEqual(release["history"][0]["upgrade_effect"], "metadata-only")


if __name__ == "__main__":
    unittest.main()
