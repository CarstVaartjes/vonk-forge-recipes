"""SGLang refuses to start gpt-oss on an attention backend without sink support.

``server_args.handle_model_specific_adjustments`` asserts that
``GptOssForCausalLM`` runs on one of the backends below; anything else (the
playbook's ``flashinfer``) crashes the server at start on a GB10 Spark.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

GPT_OSS_ATTENTION_BACKENDS = frozenset(
    {"triton", "trtllm_mha", "fa3", "fa4", "ascend", "intel_amx", "intel_xpu", "aiter"}
)
BACKEND_FLAGS = (
    "attention-backend",
    "prefill-attention-backend",
    "decode-attention-backend",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_gpt_oss(recipe: dict[str, Any]) -> bool:
    for binding in recipe.get("models", []):
        model = read(ROOT / "models" / f"{binding['model']['slug']}.json")
        if model["identity"]["family"]["slug"] == "gpt-oss":
            return True
    return False


def sglang_gpt_oss_recipes() -> list[Path]:
    return [
        path
        for path in sorted((ROOT / "recipes").glob("*.json"))
        if read(path).get("runtime", {}).get("engine") == "sglang"
        and is_gpt_oss(read(path))
    ]


def all_arguments(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    arguments = list(recipe["runtime"].get("arguments", []))
    for option in recipe.get("options", []):
        for choice in option.get("choices", []):
            arguments += choice.get("args", [])
    return arguments


class SglangGptOssAttentionBackendTests(unittest.TestCase):
    def test_guard_finds_recipes(self) -> None:
        self.assertTrue(sglang_gpt_oss_recipes())

    def test_backends_are_allowed_for_gpt_oss(self) -> None:
        for path in sglang_gpt_oss_recipes():
            arguments = all_arguments(read(path))
            with self.subTest(recipe=path.name):
                self.assertIn(
                    "attention-backend", {argument["name"] for argument in arguments}
                )
                for argument in arguments:
                    if argument["name"] in BACKEND_FLAGS:
                        self.assertIn(argument.get("value"), GPT_OSS_ATTENTION_BACKENDS)


if __name__ == "__main__":
    unittest.main()
