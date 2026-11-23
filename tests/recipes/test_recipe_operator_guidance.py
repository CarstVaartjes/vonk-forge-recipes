from __future__ import annotations

import unittest
from typing import Any

from documents import recipe_display

from recipe_tooling.qualification.documents import recipe_document


def _recipe(slug: str) -> dict[str, Any]:
    return recipe_document(slug)


def _slot(recipe: dict[str, Any], direction: str, slot_id: str) -> dict[str, Any]:
    interface = recipe["interfaces"][0]
    return next(slot for slot in interface[direction]["slots"] if slot["id"] == slot_id)


class RecipeOperatorGuidanceTests(unittest.TestCase):
    def test_wan_default_preserves_memory_headroom(self) -> None:
        recommended = _recipe("wan-2-2-ti2v-5b-comfyui-single")
        recommended_tags = set(recipe_display(recommended["identity"]["slug"]).tags)
        assert recommended_tags >= {"recommended", "default", "memory-headroom"}
        assert "physical-oom-gated" not in recommended_tags
        recommended_memory = recommended["topology"]["roles"][0]["resources"]["memory"]
        assert recommended_memory["peak_bytes"] == 78000000000
        assert recommended_memory["reserve_bytes"] == 8000000000

        for slug in (
            "wan-2-2-i2v-14b-comfyui-single",
            "wan-2-2-t2v-14b-comfyui-single",
        ):
            with self.subTest(slug=slug):
                recipe = _recipe(slug)
                tags = set(recipe_display(recipe["identity"]["slug"]).tags)
                assert tags >= {"memory-tight", "physical-oom-gated", "not-default"}
                assert "recommended" not in tags
                memory = recipe["topology"]["roles"][0]["resources"]["memory"]
                assert memory["peak_bytes"] == 118000000000
                assert memory["reserve_bytes"] == 8000000000

    def test_step1x_geometry_links_only_the_bounded_texture_stage(self) -> None:
        texture = _recipe("step1x-3d-texture-pytorch-single")
        texture_mesh = _slot(texture, "input", "mesh")
        assert texture_mesh["max_file_bytes"] == 240 * 1024 * 1024
        assert (
            "accepts-step1x-geometry"
            in recipe_display(texture["identity"]["slug"]).tags
        )

        for slug in (
            "step1x-3d-geometry-pytorch-single",
            "step1x-3d-label-geometry-pytorch-single",
        ):
            with self.subTest(slug=slug):
                geometry = _recipe(slug)
                geometry_mesh = _slot(geometry, "output", "mesh")
                assert geometry_mesh["media_types"] == texture_mesh["media_types"]
                assert geometry_mesh["extensions"] == texture_mesh["extensions"]
                assert (
                    "next-step-texture"
                    in recipe_display(geometry["identity"]["slug"]).tags
                )

    def test_ui_mate_carries_the_proposal_tags(self) -> None:
        recipe = _recipe("ui-mate-27b-vllm-single")
        tags = set(recipe_display(recipe["identity"]["slug"]).tags)
        assert tags >= {"action-proposal", "human-approval-required"}


if __name__ == "__main__":
    unittest.main()
