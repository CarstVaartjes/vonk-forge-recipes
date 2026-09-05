from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class MuseGlimmerRecipeTests(unittest.TestCase):
    def test_exact_model_selection_and_file_closure(self) -> None:
        model = load(ROOT / "models/muse-glimmer-30b-bf16-a4e59da5.json")
        recipe = load(ROOT / "recipes/muse-glimmer-30b-bf16-vllm-single.json")
        from vonk_forge_contracts import ModelDefinition
        canonical = ModelDefinition.model_validate(model).model_dump(mode="json")
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(canonical))
        self.assertEqual(model["source"]["revision"], "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
        self.assertTrue(model["files"] and all(item["sha256"] for item in model["files"]))

    def test_offline_single_spark_multimodal_contract(self) -> None:
        recipe = load(ROOT / "recipes/muse-glimmer-30b-bf16-vllm-single.json")
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        self.assertEqual(recipe["settings"]["context_tokens"]["value"], 32768)
        self.assertEqual(arguments["generation-config"], "auto")
        self.assertEqual(json.loads(arguments["limit-mm-per-prompt"]), {"image": 4, "video": 0})
        self.assertEqual(recipe["interfaces"][0]["adapter"], "openai")
        self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_runtime_and_release_are_immutable(self) -> None:
        dockerfile = (ROOT / "adapters/llm/muse-glimmer-vllm/Dockerfile").read_text()
        self.assertIn("@sha256:", dockerfile)
        self.assertNotIn("huggingface.co", dockerfile)
        recipe = load(ROOT / "recipes/muse-glimmer-30b-bf16-vllm-single.json")
        release = load(ROOT / "recipe-releases/muse-glimmer-30b-bf16-vllm-single.json")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], digest(recipe))


if __name__ == "__main__": unittest.main()
