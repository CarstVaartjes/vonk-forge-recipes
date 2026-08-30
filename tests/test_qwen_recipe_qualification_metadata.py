from __future__ import annotations

import json
import unittest
from pathlib import Path

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
                tags = set(recipe["metadata"]["tags"])
                self.assertTrue(
                    {"multimodal", "vision", "text-qualified", "vision-unproven"}
                    <= tags
                )
                self.assertIn(
                    "Current qualification proves bounded text chat only",
                    recipe["metadata"]["description"],
                )
                self.assertIn(
                    "vision serving remains unproven on Spark",
                    recipe["metadata"]["description"],
                )
                validators = recipe["validation"]["validators"]
                self.assertEqual(len(validators), 1)
                self.assertEqual(set(validators[0]["checks"]), TEXT_CHECKS)

    def test_qwen38_fp8_is_preferred_dense_default(self) -> None:
        preferred = set(_recipe("qwen3-8-27b-fp8-vllm-single")["metadata"]["tags"])
        older_dense = set(_recipe("qwen3-6-27b-vllm-single")["metadata"]["tags"])

        self.assertTrue({"preferred-dense", "default"} <= preferred)
        self.assertNotIn("superseded", preferred)
        self.assertIn("superseded", older_dense)
        self.assertTrue({"preferred-dense", "default"}.isdisjoint(older_dense))
        for alternative in (
            "qwen3-8-27b-vllm-single",
            "qwen3-8-27b-nvfp4-dspark-sglang-single",
        ):
            with self.subTest(alternative=alternative):
                tags = set(_recipe(alternative)["metadata"]["tags"])
                self.assertTrue({"preferred-dense", "default"}.isdisjoint(tags))


if __name__ == "__main__":
    unittest.main()
