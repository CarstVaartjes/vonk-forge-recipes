"""The tonyd2wild DSpark context applies its Vision-Exp port only to the vision recipe.

The port makes DeepseekV4ForCausalLM itself multimodal, and the pinned vLLM derives
multimodality from the class, so a text checkpoint built with it fails at startup while
building vision dummy inputs. Each recipe's build must match its checkpoint, and the
build-time verifier must be told which one it checks.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = "adapters/deepseek/tonyd2wild-dspark-dual"
VISION_TAGS = {"vision", "multimodal", "image", "native-vision"}
PORT_STEPS = ("patch_vision.py", "patch_registry.py", "ds4v_mm.py")


def _recipes() -> list[dict]:
    recipes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "recipes").glob("*.json"))
    ]
    return [
        recipe
        for recipe in recipes
        if recipe["execution"]["build"]["context"]["path"] == CONTEXT
    ]


class TonyDSparkVisionPortTests(unittest.TestCase):
    def test_context_serves_a_text_and_a_vision_recipe(self) -> None:
        kinds = {
            bool(VISION_TAGS & set(recipe["metadata"]["tags"])) for recipe in _recipes()
        }
        self.assertEqual(kinds, {False, True})

    def test_build_applies_the_port_exactly_for_vision_checkpoints(self) -> None:
        for recipe in _recipes():
            slug = recipe["identity"]["slug"]
            vision = bool(VISION_TAGS & set(recipe["metadata"]["tags"]))
            dockerfile = (ROOT / recipe["execution"]["build"]["dockerfile"]).read_text(
                encoding="utf-8"
            )
            run = "\n".join(
                line
                for line in dockerfile.splitlines()
                if not line.lstrip().startswith("#")
            )
            with self.subTest(slug=slug):
                for step in PORT_STEPS:
                    self.assertEqual(step in run, vision, step)
                verify = re.search(r'verify-runtime\.py"?([^;\n]*);', run)
                self.assertIsNotNone(verify)
                assert verify is not None
                self.assertEqual(verify.group(1).split(), [] if vision else ["--text"])


if __name__ == "__main__":
    unittest.main()
