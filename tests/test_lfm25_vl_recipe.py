from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = "5a414ead75d45db003906d06fb62bd5b6846cec0"
IMAGE_DIGEST = "1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c"


class Lfm25VlRecipeTests(unittest.TestCase):
    def test_checkpoint_inventory_is_exact_and_complete(self) -> None:
        version = json.loads(
            (ROOT / "model-versions/lfm2-5-vl-3b-bf16-5a414ead.json").read_text()
        )
        self.assertEqual(version["source"]["revision"], REVISION)
        self.assertEqual(version["parameters"]["total"], 3_123_483_888)
        self.assertEqual(version["limits"]["context_tokens"], 32_768)
        artifact_bytes = sum(item["download_bytes"] for item in version["artifacts"])
        self.assertEqual(artifact_bytes, 6_265_014_122)
        self.assertEqual(version["sizes"]["download_bytes"], artifact_bytes)
        paths = {item["path"] for item in version["artifacts"]}
        self.assertTrue(
            {
                "config.json",
                "model.safetensors",
                "processor_config.json",
                "tokenizer.json",
            }.issubset(paths)
        )

    def test_openai_contract_exposes_bounded_offline_multimodal_serving(self) -> None:
        recipe = json.loads(
            (ROOT / "recipes/lfm2-5-vl-3b-vllm-single.json").read_text()
        )
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        self.assertEqual(arguments["limit-mm-per-prompt"], '{"image":4}')
        self.assertEqual(arguments["allowed-local-media-path"], "/inputs")
        self.assertEqual(arguments["tool-call-parser"], "lfm2")
        self.assertIs(arguments["enable-auto-tool-choice"], True)
        mounts = {
            (item["source"], item["target"], item["read_only"])
            for item in recipe["runtime"]["security"]["mounts"]
        }
        self.assertIn(("inputs", "/inputs", True), mounts)
        self.assertEqual(recipe["interfaces"][0]["adapter"], "openai")

    def test_adapter_uses_the_pinned_cuda13_arm64_image(self) -> None:
        runtime = json.loads(
            (ROOT / "runtime-distributions/vllm-0-27-1-cuda13-arm64.json").read_text()
        )
        self.assertEqual(runtime["platform"], "linux/arm64")
        self.assertEqual(runtime["dependencies"][1]["version"], "13.0.2")
        self.assertTrue(runtime["image"].endswith(f"@sha256:{IMAGE_DIGEST}"))
        dockerfile = (
            ROOT / "adapters/liquidai/lfm25-vl-vllm/Dockerfile"
        ).read_text()
        self.assertIn(f"@sha256:{IMAGE_DIGEST}", dockerfile)
        self.assertIn(REVISION, dockerfile)


if __name__ == "__main__":
    unittest.main()
