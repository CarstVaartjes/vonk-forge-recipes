from __future__ import annotations

import json
import unittest
from pathlib import Path

from documents import recipe_display

ROOT = Path(__file__).resolve().parents[1]
QWEN_RECIPES = {
    "qwen3-5-9b-vllm-single",
    "qwen3-6-27b-vllm-single",
    "qwen3-6-35b-a3b-nvfp4-vllm-single",
    "qwen3-8-27b-fp8-vllm-single",
    "qwen3-8-27b-nvfp4-dspark-sglang-single",
    "qwen3-8-27b-vllm-single",
}
TEXT_CHECKS = {"endpoint.healthy", "chat.nonempty", "chat.max-output-64"}


def _recipe(slug: str) -> dict[str, object]:
    return json.loads((ROOT / f"recipes/{slug}.json").read_text(encoding="utf-8"))


class QwenRecipeQualificationMetadataTests(unittest.TestCase):
    def test_multimodal_capability_is_separate_from_text_only_qualification(
        self,
    ) -> None:
        for slug in sorted(QWEN_RECIPES):
            with self.subTest(recipe=slug):
                recipe = _recipe(slug)
                tags = set(recipe_display(recipe["identity"]["slug"]).tags)
                self.assertTrue(
                    {"multimodal", "vision", "text-qualified", "vision-unproven"}
                    <= tags
                )
                self.assertIn(
                    "Current qualification proves bounded text chat only",
                    recipe_display(recipe["identity"]["slug"]).description,
                )
                self.assertIn(
                    "vision serving remains unproven on Spark",
                    recipe_display(recipe["identity"]["slug"]).description,
                )
                checks = recipe["validation"]["serving"]["checks"]
                self.assertEqual(
                    {check["kind"] for check in checks},
                    {"openai.health", "openai.chat"},
                )
                self.assertIn(
                    "chat.nonempty",
                    next(
                        check["assertions"]
                        for check in checks
                        if check["kind"] == "openai.chat"
                    ),
                )

    def test_qwen38_fp8_is_preferred_dense_default(self) -> None:
        preferred = set(
            recipe_display(
                _recipe("qwen3-8-27b-fp8-vllm-single")["identity"]["slug"]
            ).tags
        )
        older_dense = set(
            recipe_display(_recipe("qwen3-6-27b-vllm-single")["identity"]["slug"]).tags
        )

        self.assertTrue({"preferred-dense", "default"} <= preferred)
        self.assertNotIn("superseded", preferred)
        self.assertIn("superseded", older_dense)
        self.assertTrue({"preferred-dense", "default"}.isdisjoint(older_dense))
        for alternative in (
            "qwen3-8-27b-vllm-single",
            "qwen3-8-27b-nvfp4-dspark-sglang-single",
        ):
            with self.subTest(alternative=alternative):
                tags = set(
                    recipe_display(_recipe(alternative)["identity"]["slug"]).tags
                )
                self.assertTrue({"preferred-dense", "default"}.isdisjoint(tags))


if __name__ == "__main__":
    unittest.main()
