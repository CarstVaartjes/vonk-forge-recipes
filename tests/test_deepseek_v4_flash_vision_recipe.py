from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model-versions/deepseek-v4-flash-vision-exp-e46e16bf.json"
PATCH = ROOT / "patch-bundles/mia-deepseek-v4-flash-vision-exp.json"
RECIPE = ROOT / "recipes/deepseek-v4-flash-vision-exp-mia-dual.json"
RELEASE = ROOT / "recipe-releases/deepseek-v4-flash-vision-exp-mia-dual.json"
TARGETS = ROOT / "model-targets/language.json"
ADAPTER = ROOT / "adapters/deepseek/mia-vllm-vision"


def document(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_digest(path: Path) -> str:
    payload = json.dumps(
        document(path), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class DeepSeekV4FlashVisionRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = document(MODEL)
        self.patch = document(PATCH)
        self.recipe = document(RECIPE)

    def test_official_checkpoint_closure_is_exact(self) -> None:
        self.assertEqual(
            self.model["source"]["revision"],
            "e46e16bf6035c6f317eb2ac7458eb0362926d402",
        )
        artifacts = self.model["artifacts"]
        self.assertEqual(len(artifacts), 84)
        self.assertEqual(sum(item["download_bytes"] for item in artifacts), 167_831_847_285)
        self.assertEqual(self.model["sizes"]["download_bytes"], 167_831_847_285)
        self.assertEqual(self.model["parameters"]["total"], 304_646_824_126)
        by_path = {item["path"]: item for item in artifacts}
        self.assertEqual(
            by_path["config.json"]["sha256"],
            "6cd841bdd6702f5e2ac34671bc78047ed80817102465525ae2a41c502abbcd75",
        )
        self.assertEqual(
            by_path["encoding/encoding_dsv4.py"]["sha256"],
            "b4bbb74bbb11a9c8ada04daa30cc7de7dba3abba08e9ade06d38b51a3d0d1701",
        )
        self.assertEqual(
            by_path["model-00001-of-00048.safetensors"]["sha256"],
            "367c971dc3cd6a042a9bec1caff508e77eabfaef1df2de3a827397ef8bbc6af3",
        )
        self.assertEqual(
            by_path["model-00048-of-00048.safetensors"]["sha256"],
            "0de99b7dd23d964b0a631e9e6f14ea1751db6d14bf0f009c6a83b725ca5f909e",
        )
        self.assertEqual(
            by_path["inference/examples/images/carrots.jpeg"]["sha256"],
            "5df896a4a07e127281c60fc957f8b3d73f4735b3258a0bf762b4383557f8fa9a",
        )
        self.assertTrue(all(item["sha256"] for item in artifacts))

    def test_mia_runtime_and_vision_overlay_are_fail_closed(self) -> None:
        self.assertEqual(
            self.patch["source"],
            {
                "repository": "https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark",
                "revision": "7963d432bfe717f2a7249a53fb4c0673c239c03e",
                "archive_sha256": "342b20c7245f8e46e39b16d679d71f2bd772e12bce2c681e52b979b16f2f730e",
            },
        )
        self.assertEqual(
            self.patch["pre_patch_tree_sha256"],
            "4a1ee8ac6eaefb8dfc9e3d15792daf9154dbaee0acab6295f1e4b95c8397c432",
        )
        self.assertEqual(
            self.patch["post_patch_tree_sha256"],
            "8079b5e2a300e12c55ec7ccb5182d89f4966de5e2dcf3d5d0917eca5164343f1",
        )
        patch_hashes = {item["path"]: item["sha256"] for item in self.patch["patches"]}
        self.assertEqual(
            patch_hashes["patches/hotfix-dsv4-vision-exp.py"],
            "882c26ed30e1e2f611bd902bd2ee63853f4eeea3eb3ca23137b2adf8b27449e8",
        )
        self.assertEqual(
            patch_hashes["patches/vision_exp/vision.py"],
            "e29feb76d7b7abfc5ae15fd152ded145d3c7c370030dfd35a0d96565112b3891",
        )
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("7963d432bfe717f2a7249a53fb4c0673c239c03e", dockerfile)
        self.assertIn("8079b5e2a300e12c55ec7ccb5182d89f4966de5e2dcf3d5d0917eca5164343f1", dockerfile)

    def test_adapter_bundle_recipe_and_patch_bundle_match(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
        archive, _files, digest = source_bundle(ADAPTER)
        self.assertEqual(len(archive), 337_920)
        self.assertEqual(digest, "48cbe006f69b229493cea87a4198440d8e553ce71883c27724575d65aff1e210")
        self.assertEqual(self.recipe["build"]["context"]["sha256"], digest)
        self.assertEqual(self.patch["source_bundle"]["sha256"], digest)
        self.assertEqual(self.recipe["model"]["content_sha256"], canonical_digest(MODEL))
        self.assertEqual(
            self.recipe["execution"]["patch_bundle"]["content_sha256"],
            canonical_digest(PATCH),
        )

    def test_dual_spark_native_vision_profile_is_exact(self) -> None:
        arguments = {
            item["name"]: item["value"] for item in self.recipe["runtime"]["arguments"]
        }
        self.assertEqual(self.recipe["topology"]["node_count"], 2)
        self.assertEqual(self.recipe["topology"]["parallelism"]["tensor"], 2)
        self.assertEqual(self.recipe["topology"]["parallelism"]["backend"], "mp")
        self.assertEqual(arguments["served-model-name"], "deepseek-v4-flash-vision-exp")
        self.assertEqual(arguments["max-model-len"], 1_048_576)
        self.assertEqual(arguments["max-num-seqs"], 6)
        self.assertEqual(arguments["max-num-batched-tokens"], 8192)
        self.assertEqual(arguments["gpu-memory-utilization"], "0.835")
        self.assertEqual(arguments["kv-cache-dtype"], "nvfp4_ds_mla")
        self.assertEqual(arguments["block-size"], 256)
        self.assertEqual(arguments["max-cudagraph-capture-size"], 42)
        self.assertEqual(json.loads(arguments["limit-mm-per-prompt"]), {"image": 8})
        self.assertEqual(
            json.loads(arguments["speculative-config"])["num_speculative_tokens"], 6
        )
        self.assertEqual(arguments["reasoning-parser"], "deepseek_v4")
        self.assertEqual(arguments["tool-call-parser"], "deepseek_v4")
        self.assertEqual(
            json.loads(arguments["default-chat-template-kwargs"]),
            {"thinking": True, "reasoning_effort": "max"},
        )
        benchmark = next(
            item
            for item in self.recipe["validation"]["benchmarks"]
            if item["name"] == "bounded-multimodal-chat"
        )
        self.assertEqual(benchmark["configuration"]["modalities"], "text,image")

    def test_target_release_and_single_spark_boundary_are_explicit(self) -> None:
        targets = document(TARGETS)["targets"]
        target = next(
            item
            for item in targets
            if item.get("catalog_model_version")
            == "deepseek-v4-flash-vision-exp-e46e16bf"
        )
        self.assertEqual(target["modality"], "multimodal")
        self.assertEqual(target["topologies"], ["distributed"])
        self.assertEqual(target["recipe_slugs"], [self.recipe["identity"]["slug"]])
        self.assertIn("No native-vision Mia single-Spark contract exists", target["notes"])
        self.assertFalse(
            any(
                path.name.startswith("deepseek-v4-flash-vision-exp")
                and path.name.endswith("-single.json")
                for path in (ROOT / "recipes").glob("*.json")
            )
        )
        release = document(RELEASE)
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            canonical_digest(RECIPE),
        )
        self.assertIn("candidate", self.recipe["metadata"]["tags"])
        self.assertNotIn("accepted", self.recipe["metadata"]["tags"])


if __name__ == "__main__":
    unittest.main()
