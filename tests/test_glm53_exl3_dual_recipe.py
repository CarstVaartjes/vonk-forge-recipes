from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm-5-3-flash-exl3-dflash2-vllm-dual.json"
ADAPTER = ROOT / "adapters/glm/mia-exl3-dflash2-dual"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Glm53Exl3DualRecipeTests(unittest.TestCase):
    def test_model_and_drafter_selections_are_closed(self) -> None:
        recipe = load(RECIPE)
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "mp")
        self.assertGreaterEqual(len(recipe["models"]), 1)
        self.assertTrue({"candidate", "executable"} <= set(recipe["metadata"]["tags"]))
        self.assertTrue(all(selection["files"] for selection in recipe["models"]))

    def test_recipe_has_bounded_speculative_profile(self) -> None:
        arguments = {item["name"]: item for item in load(RECIPE)["runtime"]["arguments"]}
        specification = json.loads(arguments["speculative-config"]["value"])
        self.assertEqual(specification["model"], "/models/drafter")
        self.assertEqual(specification["num_speculative_tokens"], 7)
        self.assertEqual(load(RECIPE)["topology"]["start_order"], ["worker", "entrypoint"])

    def test_adapter_bundle_is_pinned_and_uses_no_ssh(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        _, _, digest = tool["source_bundle"](ADAPTER)
        self.assertTrue(digest)
        text = "\n".join(path.read_text(errors="ignore") for path in ADAPTER.iterdir() if path.is_file())
        self.assertNotIn("ssh -", text.lower())

    def test_catalog_package_tracks_exact_recipe_digest(self) -> None:
        recipe = load(RECIPE)
        import hashlib
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        index = load(ROOT / "catalog-index.json")
        entry = next(item for item in index["recipes"] if item["source_path"] == f"recipes/{RECIPE.name}")
        self.assertEqual(entry["package"]["recipe_content_sha256"], digest)


if __name__ == "__main__": unittest.main()
