from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a3af5799199acdd2a4f56ac4342816abb46c12a9"
IMAGE_DIGEST = "1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c"


class Lfm25VlRecipeTests(unittest.TestCase):
    def test_checkpoint_inventory_is_exact_and_complete(self) -> None:
        version = json.loads(
            (ROOT / "model-versions/lfm2-5-vl-3b-bf16-a3af5799.json").read_text()
        )
        self.assertEqual(version["source"]["revision"], REVISION)
        self.assertEqual(version["parameters"]["total"], 3_123_483_888)
        self.assertEqual(version["limits"]["context_tokens"], 32_768)
        artifact_bytes = sum(item["download_bytes"] for item in version["artifacts"])
        self.assertEqual(artifact_bytes, 6_265_014_121)
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
        by_path = {item["path"]: item for item in version["artifacts"]}
        self.assertEqual(
            by_path["config.json"]["sha256"],
            "68ed2ea181e1fe305a43b8d5b68e9b912f1f6fd9fcb2a3c45006077ff171cb7f",
        )
        self.assertEqual(by_path["config.json"]["download_bytes"], 5_094)
        self.assertEqual(
            by_path["processor_config.json"]["sha256"],
            "ad5ce6e2a0e1acefa06d409342d4d9de7c5d064d1c4bbab7401875761c193794",
        )
        self.assertEqual(by_path["processor_config.json"]["download_bytes"], 828)

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

    def test_both_recipes_pin_the_corrected_snapshot_and_adapter_bundles(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        for slug, adapter in (
            ("lfm2-5-vl-3b-vllm-single", "adapters/liquidai/lfm25-vl-vllm"),
            (
                "lfm2-5-vl-3b-vllm028-single",
                "adapters/liquidai/lfm25-vl-vllm-028",
            ),
        ):
            recipe = json.loads((ROOT / f"recipes/{slug}.json").read_text())
            self.assertEqual(recipe["model"]["slug"], "lfm2-5-vl-3b-bf16-a3af5799")
            self.assertEqual(recipe["artifacts"][0]["revision"], REVISION)
            self.assertEqual(recipe["artifacts"][0]["download_bytes"], 6_265_014_121)
            archive, _, bundle_digest = source_bundle(ROOT / adapter)
            self.assertEqual(recipe["build"]["context"]["sha256"], bundle_digest)
            self.assertEqual(recipe["build"]["context"]["expected_bytes"], len(archive))
            release = json.loads(
                (ROOT / f"recipe-releases/{slug}.json").read_text()
            )
            self.assertEqual(release["history"][0]["upgrade_effect"], "reinstall")
            details = release["history"][0]["changes"][0]["details"]
            self.assertIn("min_tiles to 2", details)
            self.assertIn("max_position_embeddings to 32768", details)

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
