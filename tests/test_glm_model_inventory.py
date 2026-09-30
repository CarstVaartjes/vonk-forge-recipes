from __future__ import annotations

import json
import unittest
from pathlib import Path

from vonk_forge_contracts import document_sha256

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(model: dict) -> str:
    return document_sha256(model)


class GlmModelInventoryTests(unittest.TestCase):
    def test_mia_catalog_does_not_offer_an_unused_readme_only_model_revision(
        self,
    ) -> None:
        recipe = load("recipes/glm-5-3-flash-exl3-dflash2-vllm-dual.json")
        primary = next(
            selection["model"]
            for selection in recipe["models"]
            if selection["id"] == "primary"
        )
        selected = load(f"models/{primary['slug']}.json")
        payload = {
            item["path"]: item["sha256"]
            for item in selected["files"]
            if item["roles"] != ["metadata"]
        }
        for path in (ROOT / "models").glob("*.json"):
            candidate = load(str(path.relative_to(ROOT)))
            if (
                candidate["identity"]["model"] != selected["identity"]["model"]
                or candidate["identity"]["slug"] == primary["slug"]
            ):
                continue
            other_payload = {
                item["path"]: item["sha256"]
                for item in candidate["files"]
                if item["roles"] != ["metadata"]
            }
            self.assertNotEqual(
                payload,
                other_payload,
                f"{path.name} duplicates the recipe's model payload",
            )

    def test_exl3_serving_closure_does_not_prepare_an_unselected_drafter(self) -> None:
        # Wrong implementation: a target Model retains its historical drafter
        # dependency after each recipe selects a different drafter. Cache
        # preparation then resolves both and collides on shared metadata keys.
        models = {
            digest(document): document
            for path in (ROOT / "models").glob("*.json")
            for document in [load(str(path.relative_to(ROOT)))]
        }
        for path in (ROOT / "recipes").glob("glm-5-3-flash-exl3-dflash2*.json"):
            recipe = load(str(path.relative_to(ROOT)))
            selected = {item["model"]["content_sha256"] for item in recipe["models"]}
            closure = set()
            pending = list(selected)
            while pending:
                current = pending.pop()
                if current in closure:
                    continue
                closure.add(current)
                pending.extend(
                    item["content_sha256"] for item in models[current]["dependencies"]
                )
            self.assertEqual(
                closure,
                selected,
                f"{path.name} prepares an unselected companion checkpoint",
            )

    def test_current_inventory_and_recipe_select_the_calibrated_snapshot(self) -> None:
        model = load("models/glm-5-3-flash-nvfp4-caca4e6a.json")
        recipe = load("recipes/glm-5-3-flash-nvfp4-vllm-dual.json")
        self.assertEqual(
            model["source"]["revision"], "caca4e6a4ebbd66f159d3d2fc256683fd6e27177"
        )
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "ray")
        self.assertEqual(recipe["topology"]["node_count"], 2)
        files = {item["path"]: item for item in model["files"]}
        self.assertIn("model-input-scales.safetensors", files)
        self.assertIn("model.safetensors.index.json", files)

    def test_aqlm_inventory_closes_the_full_pinned_snapshot(self) -> None:
        model = load("models/glm-5-2-nvfp4-aqlm-hybrid-53e0082e.json")
        recipe = load("recipes/glm-5-2-aqlm-vllm-triple.json")
        self.assertTrue(
            any(item["path"].startswith("traces/") for item in model["files"])
        )
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "ray")
        self.assertEqual(
            len({item["id"] for item in model["files"]}), len(model["files"])
        )


if __name__ == "__main__":
    unittest.main()
