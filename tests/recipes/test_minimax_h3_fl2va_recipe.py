from __future__ import annotations

import json
import unittest
from pathlib import Path

from documents import recipe_display
from repo_root import ROOT

RECIPE_PATH = ROOT / "recipes/minimax-h3-fl2va-diffusers-single.json"
FULL_RECIPE_PATH = ROOT / "recipes/minimax-h3-diffusers-single.json"
MODEL_PATH = ROOT / "models/minimax-h3-fl2va-42ed227e.json"
ADAPTER_PATH = ROOT / "adapters/video/minimax-h3-fl2va-modular-diffusers/minimax_h3.py"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class MiniMaxH3Fl2vaRecipeTests(unittest.TestCase):
    def test_variant_is_separate_candidate_with_bounded_input_contract(self) -> None:
        recipe, full = load(RECIPE_PATH), load(FULL_RECIPE_PATH)
        assert recipe["identity"]["slug"] == "minimax-h3-fl2va-diffusers-single"
        assert {"executable", "fl2va"} <= set(
            recipe_display(recipe["identity"]["slug"]).tags
        )
        assert recipe["identity"]["slug"] != full["identity"]["slug"]
        assert recipe["interfaces"][0]["adapter"] == "video-job"

    def test_model_authority_and_filtered_selection_are_exact(self) -> None:
        model, recipe = load(MODEL_PATH), load(RECIPE_PATH)
        from vonk_forge_contracts import ModelDefinition

        canonical = ModelDefinition.model_validate(model).model_dump(mode="json")
        digest = (
            __import__("hashlib")
            .sha256(
                json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
            )
            .hexdigest()
        )
        assert recipe["models"][0]["model"]["content_sha256"] == digest
        assert model["source"]["revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
        selected = {item["file_id"] for item in recipe["models"][0]["files"]}
        assert selected == {item["id"] for item in model["files"]}

    def test_variant_has_keyframe_slots_and_no_references(self) -> None:
        slots = {
            slot["id"] for slot in load(RECIPE_PATH)["interfaces"][0]["input"]["slots"]
        }
        assert slots == {"prompt", "first-image", "last-image"}

    def test_resource_contract_uses_exact_single_spark_closure(self) -> None:
        recipe = load(RECIPE_PATH)
        role = recipe["topology"]["roles"][0]
        assert recipe["topology"]["node_count"] == 1
        assert role["resources"]["disk"]["artifact_bytes"] > 0


if __name__ == "__main__":
    unittest.main()
