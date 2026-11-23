from __future__ import annotations

import json
import unittest

from documents import recipe_display
from repo_root import ROOT

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
                tags = set(recipe_display(recipe["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
                assert {
                    "multimodal",
                    "vision",
                    "text-qualified",
                    "vision-unproven",
                } <= tags
                assert (
                    "Current qualification proves bounded text chat only"
                    in recipe_display(recipe["identity"]["slug"]).description  # pyright: ignore[reportIndexIssue]
                )
                assert (
                    "vision serving remains unproven on Spark"
                    in recipe_display(recipe["identity"]["slug"]).description  # pyright: ignore[reportIndexIssue]
                )
                checks = recipe["validation"]["serving"]["checks"]  # pyright: ignore[reportIndexIssue]
                assert {check["kind"] for check in checks} == {
                    "openai.health",
                    "openai.chat",
                }
                assert "chat.nonempty" in next(
                    check["assertions"]
                    for check in checks
                    if check["kind"] == "openai.chat"
                )

    def test_qwen38_fp8_is_preferred_dense_default(self) -> None:
        preferred = set(
            recipe_display(
                _recipe("qwen3-8-27b-fp8-vllm-single")["identity"]["slug"]  # pyright: ignore[reportIndexIssue]
            ).tags
        )
        older_dense = set(
            recipe_display(_recipe("qwen3-6-27b-vllm-single")["identity"]["slug"]).tags  # pyright: ignore[reportIndexIssue]
        )

        assert {"preferred-dense", "default"} <= preferred
        assert "superseded" not in preferred
        assert "superseded" in older_dense
        assert {"preferred-dense", "default"}.isdisjoint(older_dense)
        for alternative in (
            "qwen3-8-27b-vllm-single",
            "qwen3-8-27b-nvfp4-dspark-sglang-single",
        ):
            with self.subTest(alternative=alternative):
                tags = set(
                    recipe_display(_recipe(alternative)["identity"]["slug"]).tags  # pyright: ignore[reportIndexIssue]
                )
                assert {"preferred-dense", "default"}.isdisjoint(tags)


if __name__ == "__main__":
    unittest.main()
