from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a3af5799199acdd2a4f56ac4342816abb46c12a9"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Lfm25VlRecipeTests(unittest.TestCase):
    def test_checkpoint_inventory_is_exact_and_complete(self) -> None:
        version = load(ROOT / "models/lfm2-5-vl-3b-bf16-a3af5799.json")
        self.assertEqual(version["source"]["revision"], REVISION)
        self.assertEqual(version["parameters"]["total"], 3_123_483_888)
        paths = {item["path"] for item in version["files"]}
        self.assertTrue({"config.json", "model.safetensors", "processor_config.json", "tokenizer.json"} <= paths)
        self.assertEqual(len(paths), len(version["files"]))

    def test_openai_contract_exposes_bounded_offline_multimodal_serving(self) -> None:
        recipe = load(ROOT / "recipes/lfm2-5-vl-3b-vllm-single.json")
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        self.assertEqual(arguments["limit-mm-per-prompt"], '{"image":4}')
        self.assertEqual(arguments["allowed-local-media-path"], "/inputs")
        self.assertEqual(recipe["interfaces"][0]["adapter"], "openai")
        self.assertEqual({item["name"] for item in recipe["runtime"]["environment"]}, {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"})

    def test_both_recipes_pin_the_corrected_snapshot_and_source_bundles(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for slug, adapter in (("lfm2-5-vl-3b-vllm-single", "adapters/liquidai/lfm25-vl-vllm"), ("lfm2-5-vl-3b-vllm028-single", "adapters/liquidai/lfm25-vl-vllm-028")):
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertEqual(recipe["models"][0]["model"]["slug"], "lfm2-5-vl-3b-bf16-a3af5799")
            archive, _, bundle_digest = tool["source_bundle"](ROOT / adapter)
            self.assertTrue(archive and bundle_digest)
            self.assertEqual(recipe["execution"]["build"]["context"]["path"], adapter)

    def test_adapter_uses_pinned_cuda13_arm64_image(self) -> None:
        source = (ROOT / "adapters/liquidai/lfm25-vl-vllm/Dockerfile").read_text()
        self.assertIn(REVISION, source)
        self.assertIn("@sha256:", source)


if __name__ == "__main__": unittest.main()
