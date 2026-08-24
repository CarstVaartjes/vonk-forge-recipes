from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model-versions/muse-glimmer-30b-bf16-a4e59da5.json"
RECIPE = ROOT / "recipes/muse-glimmer-30b-bf16-vllm-single.json"
RELEASE = ROOT / "recipe-releases/muse-glimmer-30b-bf16-vllm-single.json"
RUNTIME = ROOT / "runtime-distributions/vllm-muse-glimmer-99a10304-cu130-arm64.json"
DOCKERFILE = ROOT / "adapters/llm/muse-glimmer-vllm/Dockerfile"


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class MuseGlimmerRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = json.loads(MODEL.read_text(encoding="utf-8"))
        self.recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        self.runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))

    def test_exact_model_and_runtime_closure(self) -> None:
        self.assertEqual(
            self.recipe["model"]["content_sha256"], canonical_digest(MODEL)
        )
        self.assertEqual(
            self.recipe["runtime"]["distribution"]["content_sha256"],
            canonical_digest(RUNTIME),
        )
        self.assertEqual(
            self.model["source"]["revision"],
            "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        )
        self.assertEqual(self.model["sizes"]["download_bytes"], 59_581_829_216)
        self.assertTrue(self.model["artifacts"])
        self.assertTrue(all(item["sha256"] for item in self.model["artifacts"]))

    def test_offline_single_spark_multimodal_contract(self) -> None:
        arguments = {
            item["name"]: item["value"] for item in self.recipe["runtime"]["arguments"]
        }
        self.assertEqual(arguments["max-model-len"], 32_768)
        self.assertEqual(arguments["max-num-seqs"], 1)
        self.assertEqual(arguments["generation-config"], "auto")
        self.assertEqual(arguments["reasoning-parser"], "muse_glimmer")
        self.assertEqual(arguments["tool-call-parser"], "muse_glimmer")
        self.assertEqual(arguments["allowed-local-media-path"], "/inputs")
        self.assertEqual(
            json.loads(arguments["limit-mm-per-prompt"]), {"image": 4, "video": 0}
        )
        mounts = self.recipe["runtime"]["security"]["mounts"]
        self.assertIn(
            {"source": "inputs", "target": "/inputs", "read_only": True}, mounts
        )
        self.assertFalse(self.recipe["runtime"]["security"]["host_network"])
        environment = {
            item["name"]: item["value"]
            for item in self.recipe["runtime"]["environment"]
        }
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(self.recipe["topology"]["node_count"], 1)

    def test_runtime_and_release_are_immutable(self) -> None:
        self.assertEqual(
            self.runtime["image"],
            "docker.io/vllm/vllm-openai@sha256:ae1de325b8ea670288c328460a95d3807838cd254e8e9961ea7f6a741ce2c563",
        )
        self.assertEqual(
            self.runtime["source"]["revision"],
            "99a10304dce8945119bd0b1a072297803c52a749",
        )
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn(self.runtime["image"], dockerfile)
        self.assertNotIn("huggingface.co", dockerfile)
        release = json.loads(RELEASE.read_text(encoding="utf-8"))
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            canonical_digest(RECIPE),
        )


if __name__ == "__main__":
    unittest.main()
