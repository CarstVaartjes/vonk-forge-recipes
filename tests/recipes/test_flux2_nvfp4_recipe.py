from __future__ import annotations

import hashlib
import json
import unittest
from typing import Any

from documents import recipe_display
from generated_catalog import GENERATED
from repo_root import ROOT


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def catalog_digest(recipe: dict[str, Any]) -> str:
    index = json.loads((GENERATED / "catalog-index.json").read_text(encoding="utf-8"))
    entry = next(
        item
        for item in index["recipes"]
        if item["source_path"] == f"recipes/{recipe['identity']['slug']}.json"
    )
    return entry["package"]["recipe_content_sha256"]


class Flux2NVFP4RecipeTests(unittest.TestCase):
    def test_official_nvfp4_model_and_recipe_are_exactly_bound(self) -> None:
        model = load("models/flux-2-klein-4b-nvfp4-1db2b2f7.json")
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        assert model["source"]["revision"] == "1db2b2f776c24b76f1122e5f69ab1949fc620068"
        assert model["format"]["quantization"] == "nvfp4"
        assert recipe["models"][0]["model"]["content_sha256"] == digest(model)
        assert {"executable", "nvfp4"} <= set(
            recipe_display(recipe["identity"]["slug"]).tags
        )

    def test_comfy_workflow_and_interface_remain_hash_locked(self) -> None:
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        workflow = (
            ROOT
            / "adapters/media/comfyui-image-controls/workflows/flux-2-klein-4b-controls.json"
        )
        assert (
            arguments["workflow-sha256"]
            == hashlib.sha256(workflow.read_bytes()).hexdigest()
        )
        assert recipe["interfaces"][0]["adapter"] == "image-job"
        assert recipe["topology"]["node_count"] == 1

    def test_catalog_package_binds_current_recipe(self) -> None:
        recipe = load("recipes/flux-2-klein-4b-nvfp4-comfyui-single.json")
        assert catalog_digest(recipe) == digest(recipe)


if __name__ == "__main__":
    unittest.main()
