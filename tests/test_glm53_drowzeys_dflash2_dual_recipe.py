from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual.json"
ADAPTER = ROOT / "adapters/glm/tonyd2wild-dflash2-dual"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DrowzeysGlm53Dflash2DualRecipeTests(unittest.TestCase):
    def test_model_selections_are_immutable_and_dual(self) -> None:
        recipe = load(RECIPE)
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "mp")
        self.assertEqual(len(recipe["models"]), 1)
        self.assertTrue({"candidate", "executable"} <= set(recipe["metadata"]["tags"]))
        self.assertNotIn("accepted", recipe["metadata"]["tags"])

    def test_exact_serving_profile_and_thinking_off_contract(self) -> None:
        recipe = load(RECIPE)
        arguments = {item["name"]: item for item in recipe["runtime"]["arguments"]}
        self.assertEqual(arguments["gpu-memory-utilization"]["value"], "0.85")
        self.assertEqual(json.loads(arguments["default-chat-template-kwargs"]["value"]), {"enable_thinking": False})
        self.assertEqual(recipe["topology"]["start_order"], ["worker", "entrypoint"])

    def test_adapter_bundle_is_source_pinned_and_has_no_ssh_rollout(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        _, _, bundle_digest = tool["source_bundle"](ADAPTER)
        self.assertTrue(bundle_digest)
        self.assertNotIn("ssh", "\n".join(path.read_text(errors="ignore") for path in ADAPTER.iterdir() if path.is_file()).lower())

    def test_release_tracks_recipe_content(self) -> None:
        recipe, release = load(RECIPE), load(ROOT / "recipe-releases" / f"{RECIPE.stem}.json")
        import hashlib
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(release["history"][0]["recipe_content_sha256"], digest)


if __name__ == "__main__": unittest.main()
