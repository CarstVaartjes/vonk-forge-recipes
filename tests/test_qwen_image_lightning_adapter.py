from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "adapters/image/qwen-image-lightning-diffusers/qwen_image_lightning.py"
)
LOADER = importlib.machinery.SourceFileLoader("qwen_image_lightning", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
adapter = importlib.util.module_from_spec(SPEC)
sys.dont_write_bytecode = True
LOADER.exec_module(adapter)


class QwenImageLightningAdapterTests(unittest.TestCase):
    def test_prompt_contract_reads_exactly_one_bounded_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary)
            (inputs / "prompt.txt").write_text("  draw a red fox  \n", encoding="utf-8")
            previous = adapter._INPUT_DIR
            adapter._INPUT_DIR = inputs
            try:
                self.assertEqual(adapter._prompt(), "draw a red fox")
                (inputs / "second.text").write_text("another", encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "exactly one"):
                    adapter._prompt()
            finally:
                adapter._INPUT_DIR = previous

    def test_scheduler_uses_the_published_shift_three_contract(self) -> None:
        configuration = adapter._scheduler_config()

        self.assertEqual(configuration["base_shift"], math.log(3))
        self.assertEqual(configuration["max_shift"], math.log(3))
        self.assertEqual(configuration["time_shift_type"], "exponential")
        self.assertTrue(configuration["use_dynamic_shifting"])

    def test_recipes_bind_one_exact_base_and_selected_lora(self) -> None:
        for slug in (
            "qwen-image-2512-lightning-diffusers-single",
            "qwen-image-edit-2511-lightning-diffusers-single",
        ):
            recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())

            self.assertEqual(len(recipe["models"]), 2)
            target = recipe["models"][0]
            self.assertTrue(target["files"])
            self.assertEqual(target["files"][0]["mount"]["target"], "/models/target")
            model = json.loads((ROOT / "models" / f"{target['model']['slug']}.json").read_text())
            self.assertTrue(model["files"][0]["roles"])
            self.assertNotIn("metadata-only", recipe["metadata"]["tags"])
            self.assertNotIn("non-executable", recipe["metadata"]["tags"])


if __name__ == "__main__":
    unittest.main()
