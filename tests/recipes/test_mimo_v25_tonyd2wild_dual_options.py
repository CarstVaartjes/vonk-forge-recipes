from __future__ import annotations

import json
import unittest

from repo_root import ROOT

from vonk_forge_contracts import read_recipe

RECIPE = ROOT / "recipes/mimo-v2-5-omni-nvfp4-tonyd2wild-vllm-dual.json"


class MimoV25TonyD2WildDualOptionTests(unittest.TestCase):
    def test_speculative_choices_resolve_to_one_launch_each(self) -> None:
        recipe = read_recipe(json.loads(RECIPE.read_text(encoding="utf-8")))
        expected = {
            # The default start loads no drafter (no second checkpoint pass).
            None: (None, "0"),
            "mtp1": (1, "1"),
            "mtp2": (2, "0"),
        }
        for choice, (tokens, greedy_fast) in expected.items():
            with self.subTest(choice=choice):
                selected = recipe.with_option_choices(
                    {} if choice is None else {"speculative": choice}
                )
                speculative = [
                    item.value
                    for item in selected.runtime.arguments
                    if item.name == "speculative-config"
                ]
                if tokens is None:
                    assert speculative == []
                else:
                    assert [json.loads(str(value)) for value in speculative] == [
                        {"method": "mtp", "num_speculative_tokens": tokens}
                    ]
                environment = {
                    item.name: item.value for item in selected.runtime.environment
                }
                assert environment["VLLM_MIMO_MTP1_GREEDY_FAST"] == greedy_fast
                assert "VONK_MIMO_SPEC" not in environment


if __name__ == "__main__":
    unittest.main()
