from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def canonical_digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class GlmModelInventoryTests(unittest.TestCase):
    def test_glm53_current_inventory_uses_calibrated_vllm_activation_scales(self) -> None:
        previous = load("model-versions/glm-5-3-flash-nvfp4-357b45cc.json")
        current = load("model-versions/glm-5-3-flash-nvfp4-92d8bfb9.json")
        recipe = load("recipes/glm-5-3-flash-nvfp4-vllm-dual.json")
        runtime = load("runtime-distributions/glm-5-3-flash-nvfp4-ray-dual.json")
        previous_shards = {
            item["path"]: item["sha256"]
            for item in previous["artifacts"]
            if item["path"].startswith("model-")
            and item["path"].endswith(".safetensors")
            and item["path"] != "model-input-scales.safetensors"
        }
        current_shards = {
            item["path"]: item["sha256"]
            for item in current["artifacts"]
            if item["path"].startswith("model-")
            and item["path"].endswith(".safetensors")
            and item["path"] != "model-input-scales.safetensors"
        }
        artifacts = {item["path"]: item for item in current["artifacts"]}
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }

        self.assertEqual(current["source"]["revision"], "92d8bfb91c19ceb6fb530dfb538a3a24eceb6ef7")
        self.assertEqual(len(current["artifacts"]), 131)
        self.assertEqual(current["sizes"]["download_bytes"], 194_701_810_857)
        self.assertEqual(previous_shards, current_shards)
        self.assertEqual(
            artifacts["model-input-scales.safetensors"]["sha256"],
            "b812b4d7df15dedb06e74defa6380fecba61d45b8c117fb42eda1443430d2a4a",
        )
        self.assertEqual(
            artifacts["model.safetensors.index.json"]["sha256"],
            "0a5cb10ccc9bfdfe7a4e857e69fb6eea229cc98343986db27ca95e4266e1a2d9",
        )
        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(current))
        self.assertEqual(recipe["artifacts"][0]["revision"], current["source"]["revision"])
        self.assertEqual(arguments["tool-call-parser"], "glm47")
        self.assertEqual(arguments["reasoning-parser"], "deepseek_r1")
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "ray")
        self.assertEqual(
            runtime["source"]["revision"],
            "aed98a13ca75140d2691cc5c651ea5817d9a3e44",
        )
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/"
            "tree/aed98a13ca75140d2691cc5c651ea5817d9a3e44",
        )

    def test_aqlm_inventory_closes_the_full_pinned_snapshot(self) -> None:
        model = load("model-versions/glm-5-2-nvfp4-aqlm-hybrid-53e0082e.json")
        recipe = load("recipes/glm-5-2-aqlm-vllm-triple.json")
        runtime = load("runtime-distributions/glm-5-2-aqlm-triple-dspark.json")
        artifacts = model["artifacts"]
        traces = [item for item in artifacts if item["path"].startswith("traces/")]

        self.assertEqual(len(artifacts), 751)
        self.assertEqual(len(traces), 603)
        self.assertEqual(sum(item["download_bytes"] for item in traces), 83_970_453)
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            292_599_148_533,
        )
        self.assertEqual(model["sizes"]["download_bytes"], 292_599_148_533)
        self.assertEqual(recipe["artifacts"][0]["download_bytes"], 292_599_148_533)
        self.assertEqual(len({item["id"] for item in artifacts}), len(artifacts))
        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(model))
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "ray")
        self.assertEqual(
            runtime["capabilities"]["distributed_vllm"]["mechanism"],
            "vllm-ray",
        )

    def test_gated_abliterated_inventory_fails_closed_without_fake_artifact(self) -> None:
        model = load(
            "model-versions/glm-5-3-flash-nvfp4-abliterated-d7f8afa8.json"
        )
        recipe = load(
            "recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json"
        )
        artifacts = model["artifacts"]

        self.assertEqual(model["availability"], "withdrawn")
        self.assertEqual(len(artifacts), 126)
        self.assertNotIn("snapshot", {item["path"] for item in artifacts})
        self.assertEqual(len({item["id"] for item in artifacts}), len(artifacts))
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            194_692_710_561,
        )
        self.assertEqual(model["sizes"]["download_bytes"], 194_692_710_561)
        self.assertEqual(
            recipe["artifacts"][0]["download_bytes"]
            - model["sizes"]["download_bytes"],
            50_928,
        )
        self.assertIn("13 Git files", model["metadata"]["description"])
        self.assertIn("HTTP 401", model["metadata"]["description"])
        self.assertIn("inventory-blocked", recipe["metadata"]["tags"])
        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(model))


if __name__ == "__main__":
    unittest.main()
