from __future__ import annotations

import json
import unittest
from pathlib import Path

from documents import recipe_display

ROOT = Path(__file__).resolve().parents[1]


def _recipe(slug: str) -> dict[str, object]:
    return json.loads((ROOT / "recipes" / f"{slug}.json").read_text(encoding="utf-8"))


def _slot(recipe: dict[str, object], direction: str, slot_id: str) -> dict[str, object]:
    interface = recipe["interfaces"][0]  # pyright: ignore[reportIndexIssue]
    return next(slot for slot in interface[direction]["slots"] if slot["id"] == slot_id)


class RecipeOperatorGuidanceTests(unittest.TestCase):
    def test_wan_default_preserves_memory_headroom(self) -> None:
        recommended = _recipe("wan-2-2-ti2v-5b-comfyui-single")
        recommended_tags = set(recipe_display(recommended["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        self.assertGreaterEqual(
            recommended_tags, {"recommended", "default", "memory-headroom"}
        )
        self.assertNotIn("physical-oom-gated", recommended_tags)
        recommended_memory = recommended["topology"]["roles"][0]["resources"]["memory"]  # pyright: ignore[reportIndexIssue]
        self.assertEqual(recommended_memory["peak_bytes"], 78_000_000_000)
        self.assertEqual(recommended_memory["reserve_bytes"], 8_000_000_000)

        for slug in (
            "wan-2-2-i2v-14b-comfyui-single",
            "wan-2-2-t2v-14b-comfyui-single",
        ):
            with self.subTest(slug=slug):
                recipe = _recipe(slug)
                tags = set(recipe_display(recipe["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
                self.assertGreaterEqual(
                    tags, {"memory-tight", "physical-oom-gated", "not-default"}
                )
                self.assertNotIn("recommended", tags)
                memory = recipe["topology"]["roles"][0]["resources"]["memory"]  # pyright: ignore[reportIndexIssue]
                self.assertEqual(memory["peak_bytes"], 118_000_000_000)
                self.assertEqual(memory["reserve_bytes"], 8_000_000_000)

    def test_step1x_geometry_links_only_the_bounded_texture_stage(self) -> None:
        texture = _recipe("step1x-3d-texture-pytorch-single")
        texture_mesh = _slot(texture, "input", "mesh")
        self.assertEqual(texture_mesh["max_file_bytes"], 240 * 1024 * 1024)
        self.assertIn(
            "accepts-step1x-geometry",
            recipe_display(texture["identity"]["slug"]).tags,  # pyright: ignore[reportIndexIssue]
        )

        for slug in (
            "step1x-3d-geometry-pytorch-single",
            "step1x-3d-label-geometry-pytorch-single",
        ):
            with self.subTest(slug=slug):
                geometry = _recipe(slug)
                geometry_mesh = _slot(geometry, "output", "mesh")
                self.assertEqual(
                    geometry_mesh["media_types"], texture_mesh["media_types"]
                )
                self.assertEqual(
                    geometry_mesh["extensions"], texture_mesh["extensions"]
                )
                self.assertIn(
                    "next-step-texture",
                    recipe_display(geometry["identity"]["slug"]).tags,  # pyright: ignore[reportIndexIssue]
                )

    def test_ui_mate_carries_the_proposal_tags(self) -> None:
        recipe = _recipe("ui-mate-27b-vllm-single")
        tags = set(recipe_display(recipe["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
        self.assertGreaterEqual(tags, {"action-proposal", "human-approval-required"})


if __name__ == "__main__":
    unittest.main()
