from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from vonk_forge_contracts import read_recipe

ROOT = Path(__file__).resolve().parents[1]
RECIPE_NAMES = (
    "qwen3-6-35b-a3b-fp8-eugr-vllm-single",
    "qwen3-6-35b-a3b-fp8-eugr-vllm-dual",
    "qwen3-6-35b-a3b-fp8-dflash-eugr-vllm-single",
    "qwen3-6-35b-a3b-fp8-dflash-eugr-vllm-dual",
)
MODEL = ROOT / "models/qwen3-6-35b-a3b-fp8-95a723d0.json"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def argument_map(recipe: Any) -> dict[str, Any]:
    return {item.name: item.value for item in recipe.runtime.arguments}


class Qwen36Fp8OptionTests(unittest.TestCase):
    def test_primary_checkpoint_contains_the_native_mtp_head(self) -> None:
        model = read(MODEL)
        self.assertEqual(
            model["source"]["revision"],
            "95a723d08a9490559dae23d0cff1d9466213d989",
        )
        self.assertIn("mtp.safetensors", {item["path"] for item in model["files"]})

    def test_every_rank_and_interacting_choice_compiles_without_conflicting_modes(
        self,
    ) -> None:
        for name in RECIPE_NAMES:
            raw = read(ROOT / "recipes" / f"{name}.json")
            recipe = read_recipe(raw)
            is_dflash = "-dflash-" in name
            baseline = recipe.runtime
            default_speculation = "dflash" if is_dflash else "off"
            self.assertEqual(
                recipe.resolve_options(),
                {"speculation": default_speculation, "modality": "multimodal"},
            )
            self.assertEqual(
                recipe.with_option_choices(recipe.resolve_options()).runtime,
                baseline,
                f"default option selection changed the current runtime for {name}",
            )

            for speculation in (
                ("dflash", "native-mtp") if is_dflash else ("off", "native-mtp")
            ):
                for modality in ("multimodal", "text-only"):
                    with self.subTest(
                        recipe=name, speculation=speculation, modality=modality
                    ):
                        selected = recipe.with_option_choices(
                            {"speculation": speculation, "modality": modality}
                        )
                        arguments = argument_map(selected)
                        speculative = [
                            item
                            for item in selected.runtime.arguments
                            if item.name == "speculative-config"
                        ]
                        self.assertLessEqual(len(speculative), 1)
                        if speculation == "native-mtp":
                            self.assertEqual(
                                json.loads(str(arguments["speculative-config"])),
                                {
                                    "method": "qwen3_next_mtp",
                                    "num_speculative_tokens": 2,
                                },
                            )
                        elif speculation == "dflash":
                            self.assertEqual(
                                json.loads(str(arguments["speculative-config"])),
                                {
                                    "method": "dflash",
                                    "model": "/models/drafter",
                                    "num_speculative_tokens": 15,
                                },
                            )
                        else:
                            self.assertNotIn("speculative-config", arguments)
                        self.assertEqual(
                            arguments.get("language-model-only"),
                            modality == "text-only" or None,
                        )
                        self.assertEqual(selected.topology, recipe.topology)
                        self.assertEqual(selected.models, recipe.models)
                        self.assertEqual(selected.execution, recipe.execution)
                        self.assertEqual(selected.interfaces, recipe.interfaces)


if __name__ == "__main__":
    unittest.main()
