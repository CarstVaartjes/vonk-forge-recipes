from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _recipe(slug: str) -> dict[str, object]:
    return json.loads((ROOT / "recipes" / f"{slug}.json").read_text(encoding="utf-8"))


def _slot(recipe: dict[str, object], direction: str, slot_id: str) -> dict[str, object]:
    interface = recipe["interfaces"][0]
    return next(slot for slot in interface[direction]["slots"] if slot["id"] == slot_id)


class RecipeOperatorGuidanceTests(unittest.TestCase):
    def test_wan_default_preserves_memory_headroom(self) -> None:
        recommended = _recipe("wan-2-2-ti2v-5b-comfyui-single")
        recommended_tags = set(recommended["metadata"]["tags"])
        self.assertGreaterEqual(
            recommended_tags, {"recommended", "default", "memory-headroom"}
        )
        self.assertNotIn("physical-oom-gated", recommended_tags)
        recommended_memory = recommended["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(recommended_memory["startup_peak_bytes"], 78_000_000_000)
        self.assertEqual(recommended_memory["system_reserve_bytes"], 8_000_000_000)

        for slug in (
            "wan-2-2-i2v-14b-comfyui-single",
            "wan-2-2-t2v-14b-comfyui-single",
        ):
            with self.subTest(slug=slug):
                recipe = _recipe(slug)
                tags = set(recipe["metadata"]["tags"])
                self.assertGreaterEqual(
                    tags, {"memory-tight", "physical-oom-gated", "not-default"}
                )
                self.assertNotIn("recommended", tags)
                memory = recipe["topology"]["roles"][0]["resources"]["memory"]
                self.assertEqual(memory["startup_peak_bytes"], 118_000_000_000)
                self.assertEqual(memory["system_reserve_bytes"], 8_000_000_000)

    def test_native_three_d_candidates_publish_their_acceptance_gates(self) -> None:
        for slug in ("skintokens-pytorch-single", "triposg-pytorch-single"):
            with self.subTest(slug=slug):
                recipe = _recipe(slug)
                tags = set(recipe["metadata"]["tags"])
                self.assertGreaterEqual(
                    tags,
                    {
                        "executable",
                        "candidate",
                        "build-unvalidated",
                        "native-build-required",
                        "physical-acceptance-required",
                    },
                )
                description = recipe["metadata"]["description"].lower()
                self.assertIn("not accepted", description)
                self.assertIn("build", description)
                self.assertIn("physical", description)
                self.assertIn("canary", description)

    def test_step1x_geometry_links_only_the_bounded_texture_stage(self) -> None:
        texture = _recipe("step1x-3d-texture-pytorch-single")
        texture_mesh = _slot(texture, "input", "mesh")
        self.assertEqual(texture_mesh["max_file_bytes"], 240 * 1024 * 1024)
        self.assertIn("accepts-step1x-geometry", texture["metadata"]["tags"])

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
                description = geometry["metadata"]["description"]
                self.assertIn(
                    "vonk-forge/step1x-3d-texture-pytorch-single", description
                )
                self.assertIn("up to 240 MiB", description)
                self.assertIn("next-step-texture", geometry["metadata"]["tags"])

        texture_description = texture["metadata"]["description"]
        self.assertIn(
            "vonk-forge/step1x-3d-geometry-pytorch-single", texture_description
        )
        self.assertIn(
            "vonk-forge/step1x-3d-label-geometry-pytorch-single",
            texture_description,
        )
        self.assertIn("up to 240 MiB", texture_description)

    def test_ui_mate_is_described_as_a_non_actuating_proposal_interface(self) -> None:
        recipe = _recipe("ui-mate-27b-vllm-single")
        tags = set(recipe["metadata"]["tags"])
        self.assertGreaterEqual(tags, {"action-proposal", "human-approval-required"})
        description = recipe["metadata"]["description"].lower()
        self.assertIn("non-actuating", description)
        self.assertIn("approve", description)
        self.assertIn("execute actions externally", description)
        benchmark = recipe["validation"]["benchmarks"][0]
        self.assertEqual(benchmark["name"], "bounded-action-proposal")


if __name__ == "__main__":
    unittest.main()
