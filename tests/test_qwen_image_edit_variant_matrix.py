from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDIT_VARIANTS = (
    "qwen-image-edit-2511-comfyui-single",
    "qwen-image-edit-2511-diffusers-single",
    "qwen-image-edit-2511-fp8mixed-comfyui-single",
    "qwen-image-edit-2511-int8-convrot-comfyui-single",
    "qwen-image-edit-2511-lightning-diffusers-single",
)
LAYERED = "qwen-image-layered-diffusers-single"
EXPECTED_ENVELOPES = {
    "qwen-image-edit-2511-comfyui-single": (126_000_000_000, 135_499_508_486),
    "qwen-image-edit-2511-diffusers-single": (126_000_000_000, 149_720_463_453),
    "qwen-image-edit-2511-fp8mixed-comfyui-single": (
        108_000_000_000,
        95_172_239_743,
    ),
    "qwen-image-edit-2511-int8-convrot-comfyui-single": (
        108_000_000_000,
        95_137_560_750,
    ),
    "qwen-image-edit-2511-lightning-diffusers-single": (
        126_000_000_000,
        150_570_071_749,
    ),
    "qwen-image-layered-diffusers-single": (126_000_000_000, 149_720_484_610),
}


def load(directory: str, slug: str) -> dict[str, object]:
    return json.loads((ROOT / directory / f"{slug}.json").read_text(encoding="utf-8"))


class QwenImageEditVariantMatrixTests(unittest.TestCase):
    def test_general_edit_variants_have_clear_roles_without_a_premature_default(self) -> None:
        recipes = {slug: load("recipes", slug) for slug in EDIT_VARIANTS}
        self.assertFalse(
            any(
                {"default", "recommended"}.intersection(recipe["metadata"]["tags"])
                for recipe in recipes.values()
            )
        )
        self.assertIn(
            "memory-efficient",
            recipes["qwen-image-edit-2511-fp8mixed-comfyui-single"]["metadata"][
                "tags"
            ],
        )
        int8_tags = recipes[
            "qwen-image-edit-2511-int8-convrot-comfyui-single"
        ]["metadata"]["tags"]
        self.assertTrue({"memory-efficient", "int8", "convrot"} <= set(int8_tags))
        self.assertIn(
            "fast",
            recipes["qwen-image-edit-2511-lightning-diffusers-single"]["metadata"][
                "tags"
            ],
        )
        for slug in (
            "qwen-image-edit-2511-comfyui-single",
            "qwen-image-edit-2511-diffusers-single",
        ):
            self.assertIn("quality-reference", recipes[slug]["metadata"]["tags"])

        for recipe in recipes.values():
            tags = set(recipe["metadata"]["tags"])
            self.assertTrue({"candidate", "executable", "single", "spark"} <= tags)
            self.assertNotIn("superseded", tags)

    def test_declared_envelopes_fit_one_128_gb_spark(self) -> None:
        for slug, expected in EXPECTED_ENVELOPES.items():
            with self.subTest(slug=slug):
                recipe = load("recipes", slug)
                resources = recipe["topology"]["roles"][0]["resources"]
                memory = resources["memory"]
                memory_envelope = max(
                    memory["startup_peak_bytes"],
                    memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
                ) + memory["system_reserve_bytes"]
                disk_envelope = sum(resources["disk"].values())
                self.assertEqual((memory_envelope, disk_envelope), expected)
                self.assertLessEqual(memory_envelope, 128_000_000_000)

    def test_edit_input_and_output_contracts_match_every_general_variant(self) -> None:
        for slug in EDIT_VARIANTS:
            with self.subTest(slug=slug):
                recipe = load("recipes", slug)
                interface = recipe["interfaces"][0]
                self.assertEqual(interface["adapter"], "image-job")
                inputs = {slot["id"]: slot for slot in interface["input"]["slots"]}
                self.assertEqual(set(inputs), {"prompt", "image"})
                self.assertEqual(
                    (inputs["prompt"]["min_files"], inputs["prompt"]["max_files"]),
                    (1, 1),
                )
                self.assertEqual(
                    (inputs["image"]["min_files"], inputs["image"]["max_files"]),
                    (1, 2),
                )
                outputs = interface["output"]["slots"]
                self.assertEqual(len(outputs), 1)
                self.assertEqual(
                    (outputs[0]["min_files"], outputs[0]["max_files"]),
                    (1, 1),
                )
                self.assertEqual(outputs[0]["media_types"], ["image/png"])
                self.assertEqual(
                    recipe["validation"]["validators"],
                    [{"interface": "image-job", "checks": ["artifact.mime.image-png"]}],
                )

    def test_layered_is_a_separate_exact_four_output_task(self) -> None:
        recipe = load("recipes", LAYERED)
        tags = set(recipe["metadata"]["tags"])
        self.assertIn("layer-decomposition", tags)
        self.assertNotIn("recommended", tags)
        interface = recipe["interfaces"][0]
        self.assertEqual(interface["adapter"], "artifact-job")
        image_input = interface["input"]["slots"][0]
        self.assertEqual((image_input["min_files"], image_input["max_files"]), (1, 1))
        layers = interface["output"]["slots"][0]
        self.assertEqual((layers["min_files"], layers["max_files"]), (4, 4))
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        self.assertEqual(
            (arguments["layers"], arguments["resolution"], arguments["num-inference-steps"]),
            (4, 640, 50),
        )

    def test_all_bound_model_versions_are_active_and_current_runtime_families(self) -> None:
        for slug in (*EDIT_VARIANTS, LAYERED):
            with self.subTest(slug=slug):
                recipe = load("recipes", slug)
                model = load("model-versions", recipe["model"]["slug"])
                self.assertEqual(model["availability"], "active")
                runtime = recipe["runtime"]["distribution"]["slug"]
                self.assertIn(
                    runtime,
                    {
                        "comfyui-0-33-4-cuda13-arm64",
                        "diffusers-0-40-0-cuda13-arm64",
                    },
                )


if __name__ == "__main__":
    unittest.main()
