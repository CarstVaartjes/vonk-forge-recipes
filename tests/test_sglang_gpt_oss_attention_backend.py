"""SGLang refuses to start gpt-oss on an attention backend without sink support.

``server_args.handle_model_specific_adjustments`` asserts that
``GptOssForCausalLM`` runs on one of the backends below; anything else (the
playbook's ``flashinfer``) crashes the server at start on a GB10 Spark.
"""

from __future__ import annotations

import unittest

from gpt_oss_recipes import all_arguments, gpt_oss_recipes, read

GPT_OSS_ATTENTION_BACKENDS = frozenset(
    {"triton", "trtllm_mha", "fa3", "fa4", "ascend", "intel_amx", "intel_xpu", "aiter"}
)
BACKEND_FLAGS = (
    "attention-backend",
    "prefill-attention-backend",
    "decode-attention-backend",
)


class SglangGptOssAttentionBackendTests(unittest.TestCase):
    def test_guard_finds_recipes(self) -> None:
        self.assertTrue(gpt_oss_recipes("sglang"))

    def test_backends_are_allowed_for_gpt_oss(self) -> None:
        for path in gpt_oss_recipes("sglang"):
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
