from __future__ import annotations

import hashlib
import json
import runpy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import ModelDefinition, RecipeDefinition  # noqa: E402
from vonk_forge_contracts.resolver import validate_recipe_models  # noqa: E402

RECIPES = {
    name: ROOT / "recipes" / filename
    for name, filename in {
        "nano": "nemotron-3-nano-30b-a3b-vllm-single.json",
        "omni": "nemotron-3-nano-omni-30b-a3b-vllm-single.json",
        "super": "nemotron-3-super-120b-a12b-vllm-single.json",
    }.items()
}
REVISIONS = {
    "nano": "ce1b118ae66ec705d02c241525192832eb045fd3",
    "omni": "16993199e436da4ba75ddc410855f87e0d996ee6",
    "super": "ff433f5493e25d631c9f12b5d55c674229923d02",
}


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def recipe_model(recipe: dict[str, object], index: int = 0) -> dict[str, object]:
    return read(ROOT / "models" / (recipe["models"][index]["model"]["slug"] + ".json"))  # type: ignore[index]


def arguments(recipe: dict[str, object]) -> dict[str, object]:
    return {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}  # type: ignore[index]


class NemotronExecutableRecipeTests(unittest.TestCase):
    def test_each_profile_resolves_exact_model_snapshots(self) -> None:
        models = [ModelDefinition.model_validate(read(path)) for path in sorted((ROOT / "models").glob("*.json"))]
        for name, path in RECIPES.items():
            with self.subTest(profile=name):
                recipe = RecipeDefinition.model_validate(read(path))
                validate_recipe_models(recipe, models)
                model = recipe_model(read(path))
                self.assertEqual(model["source"]["revision"], REVISIONS[name])
                self.assertEqual(recipe.runtime.engine, "vllm")
                self.assertEqual(recipe.topology.node_count, 1)
                self.assertIn("candidate", recipe.metadata.tags)

    def test_custom_reasoning_parser_is_selected_from_model_manifest(self) -> None:
        expected = {"nano": "nano_v3_reasoning_parser.py", "super": "super_v3_reasoning_parser.py"}
        for name, filename in expected.items():
            with self.subTest(profile=name):
                recipe = read(RECIPES[name])
                args = arguments(recipe)
                self.assertEqual(args["reasoning-parser-plugin"].split("/")[-1], filename)
                model = recipe_model(recipe)
                selected = {item["file_id"] for item in recipe["models"][0]["files"]}  # type: ignore[index]
                manifest = next(item for item in model["files"] if item["path"] == filename)  # type: ignore[index]
                self.assertIn(manifest["id"], selected)
                self.assertIn("runtime", manifest["roles"])

    def test_super_uses_fp4_mamba_and_mtp_contract(self) -> None:
        recipe = read(RECIPES["super"])
        args = arguments(recipe)
        self.assertEqual(args["quantization"], "modelopt_fp4")
        self.assertEqual(args["moe-backend"], "marlin")
        self.assertEqual(args["mamba-ssm-cache-dtype"], "float16")
        self.assertEqual(json.loads(args["speculative-config"]), {"method": "mtp", "num_speculative_tokens": 3, "model": "/models/drafter", "moe_backend": "triton"})
        self.assertEqual(len(recipe["models"]), 2)
        drafter = recipe["models"][1]["files"][0]  # type: ignore[index]
        self.assertEqual(drafter["mount"]["target"], "/models/drafter")

    def test_omni_is_explicitly_text_only(self) -> None:
        recipe = read(RECIPES["omni"])
        self.assertTrue(arguments(recipe)["language-model-only"])
        self.assertTrue({"text", "text-only"} <= set(recipe["metadata"]["tags"]))
        self.assertEqual(recipe["interfaces"][0]["adapter"], "openai")
        model = recipe_model(recipe)
        self.assertEqual(model["modalities"], ["text"])
        self.assertFalse(any(fact["support"] == "supported" and fact["capability"] in {"image-understanding", "audio-understanding", "video-understanding"} for fact in model["capabilities"]["facts"]))

    def test_source_builds_are_offline_and_digest_pinned(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for path in RECIPES.values():
            recipe = read(path)
            build = recipe["execution"]["build"]
            context = ROOT / build["context"]["path"]
            archive, _, digest = tool["source_bundle"](context)
            self.assertEqual(recipe["execution"]["mode"], "build")
            self.assertEqual(build["network"]["mode"], "none")
            self.assertRegex(digest, r"^[a-f0-9]{64}$")
            dockerfile = (ROOT / build["dockerfile"]).read_text(encoding="utf-8")
            self.assertNotIn("pip install", dockerfile)
            self.assertNotIn("apt-get", dockerfile)


if __name__ == "__main__":
    unittest.main()
