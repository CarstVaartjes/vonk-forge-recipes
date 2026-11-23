from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
from documents import recipe_display
from repo_root import ROOT

SCRIPT = ROOT / "adapters/image/qwen-image-lightning-diffusers/qwen_image_lightning.py"
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
            adapter._INPUT_DIR = inputs  # pyright: ignore[reportAttributeAccessIssue]
            try:
                assert adapter._prompt() == "draw a red fox"
                (inputs / "second.text").write_text("another", encoding="utf-8")
                with pytest.raises(SystemExit, match="exactly one"):
                    adapter._prompt()
            finally:
                adapter._INPUT_DIR = previous  # pyright: ignore[reportAttributeAccessIssue]

    def test_scheduler_uses_the_published_shift_three_contract(self) -> None:
        configuration = adapter._scheduler_config()

        assert configuration["base_shift"] == math.log(3)
        assert configuration["max_shift"] == math.log(3)
        assert configuration["time_shift_type"] == "exponential"
        assert configuration["use_dynamic_shifting"]

    def test_recipes_bind_one_exact_base_and_selected_lora(self) -> None:
        for slug in (
            "qwen-image-2512-lightning-diffusers-single",
            "qwen-image-edit-2511-lightning-diffusers-single",
        ):
            recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())

            assert len(recipe["models"]) == 2
            target = recipe["models"][0]
            assert target["files"]
            assert target["files"][0]["mount"]["target"] == "/models/target"
            model = json.loads(
                (ROOT / "models" / f"{target['model']['slug']}.json").read_text()
            )
            assert model["files"][0]["roles"]
            assert (
                "metadata-only" not in recipe_display(recipe["identity"]["slug"]).tags
            )
            assert (
                "non-executable" not in recipe_display(recipe["identity"]["slug"]).tags
            )


if __name__ == "__main__":
    unittest.main()
