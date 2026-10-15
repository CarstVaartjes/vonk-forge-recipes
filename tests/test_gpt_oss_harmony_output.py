"""gpt-oss recipes must turn harmony output into a valid OpenAI response offline.

SGLang keeps gpt-oss's harmony special tokens in the generated text and only
its ``gpt-oss`` reasoning and tool-call parsers strip the channel markup; without
them ``<|channel|>`` leaks into ``message.content``. vLLM and TensorRT-LLM parse
harmony through ``openai_harmony``, which downloads its tiktoken vocab unless
``TIKTOKEN_ENCODINGS_BASE`` names a local directory; workloads run offline, so
the encodings must be baked into the image there.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from gpt_oss_recipes import ROOT, all_arguments, gpt_oss_recipes, read

HARMONY_ENCODINGS = ("o200k_base.tiktoken", "cl100k_base.tiktoken")


def environment(recipe: dict[str, Any]) -> dict[str, str]:
    return {item["name"]: item["value"] for item in recipe["runtime"]["environment"]}


def copied_sources(recipe: dict[str, Any], destination: str) -> list[Path]:
    """Build-context directories the Dockerfile copies to ``destination``."""

    build = recipe["execution"]["build"]
    context = ROOT / build["context"]["path"]
    sources = []
    for line in (ROOT / build["dockerfile"]).read_text(encoding="utf-8").splitlines():
        words = [word for word in line.split() if not word.startswith("--")]
        if (
            len(words) == 3
            and words[0] == "COPY"
            and words[2].rstrip("/") == destination.rstrip("/")
        ):
            sources.append(context / words[1])
    return sources


class GptOssHarmonyOutputTests(unittest.TestCase):
    def test_guard_finds_recipes(self) -> None:
        self.assertTrue(gpt_oss_recipes("sglang"))
        self.assertTrue(gpt_oss_recipes("vllm", "tensorrt-llm"))

    def test_sglang_parses_harmony_output(self) -> None:
        for path in gpt_oss_recipes("sglang"):
            arguments = {
                argument["name"]: argument.get("value")
                for argument in all_arguments(read(path))
            }
            with self.subTest(recipe=path.name):
                self.assertEqual(arguments.get("reasoning-parser"), "gpt-oss")
                self.assertEqual(arguments.get("tool-call-parser"), "gpt-oss")

    def test_harmony_vocab_is_baked_into_the_image(self) -> None:
        for path in gpt_oss_recipes("vllm", "tensorrt-llm"):
            recipe = read(path)
            with self.subTest(recipe=path.name):
                base = environment(recipe).get("TIKTOKEN_ENCODINGS_BASE")
                self.assertIsNotNone(base)
                sources = copied_sources(recipe, base or "")
                self.assertEqual(len(sources), 1)
                for name in HARMONY_ENCODINGS:
                    self.assertTrue((sources[0] / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
