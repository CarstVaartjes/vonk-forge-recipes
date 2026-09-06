from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {"glm-5-2-aqlm-vllm-triple": 3, "glm-5-2-quanttrio-vllm-four": 4, "glm-5-3-flash-nvfp4-vllm-four": 4, "inkling-small-nvfp4-sglang-dual": 2}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class DistributedRecipeAvailabilityTests(unittest.TestCase):
    def test_larger_topologies_and_candidate_metadata_are_explicit(self) -> None:
        for slug, nodes in RECIPES.items():
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertEqual(recipe["topology"]["node_count"], nodes)
            self.assertTrue({"candidate", "executable"} <= set(recipe["metadata"]["tags"]))
            self.assertTrue(recipe["models"])

    def test_per_node_disk_envelopes_are_bounded(self) -> None:
        for slug in RECIPES:
            disk = load(ROOT / "recipes" / f"{slug}.json")["topology"]["roles"][0]["resources"]["disk"]
            self.assertGreater(disk["artifact_bytes"], 0)
            self.assertGreater(disk["staging_bytes"], 0)

    def test_glm_tp4_tracks_current_public_source_profile(self) -> None:
        recipe = load(ROOT / "recipes/glm-5-3-flash-nvfp4-vllm-four.json")
        adapter = ROOT / "adapters/glm/tonyd2wild-glm53-tp4-current"
        dockerfile = (adapter / "Dockerfile").read_text()
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        self.assertEqual(recipe["execution"]["build"]["context"]["path"], "adapters/glm/tonyd2wild-glm53-tp4-current")
        self.assertEqual(recipe["provenance"]["source_reference"], "https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/tree/8fd2fcd27c04c7fa93e770000b818657f338875d")
        self.assertEqual(recipe["execution"]["build"]["base_image"]["digest"], "905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce")
        self.assertEqual(arguments["tensor-parallel-size"], 4)
        self.assertEqual(arguments["max-model-len"], 1048576)
        self.assertEqual(arguments["max-num-seqs"], 64)
        self.assertEqual(arguments["max-num-batched-tokens"], 16384)
        self.assertEqual(arguments["kv-cache-memory"], 25769803776)
        self.assertEqual(arguments["speculative-config"], '{"method":"dflash","model":"/models/drafter","num_speculative_tokens":7}')
        self.assertFalse(arguments["enforce-eager"])
        self.assertEqual(arguments["compilation-config"], '{"cudagraph_mode":"FULL_AND_PIECEWISE"}')
        drafter = next(
            selection
            for selection in recipe["models"]
            if selection["id"] == "dependency-glm-5-3-flash-dflash2-bf582e4e"
        )
        self.assertEqual(
            {file["file_id"] for file in drafter["files"]},
            {
                "metadata-70e0ed421d65-618cd5b83d",
                "readme-2014434d3e36-b335630551",
                "dflash2-figure-6d8dcc9a9472-e520c8f797",
                "config-c4aeac010119-587cb980af",
                "model-b038e1d9d1e7-9d75c1098f",
            },
        )
        self.assertEqual(
            {file["mount"]["target"] for file in drafter["files"]},
            {"/models/drafter"},
        )
        self.assertIn('ai.vonkforge.runtime-interface="v1"', dockerfile)
        self.assertIn("COPY overlay-dflash2/qwen3_dflash2.py", dockerfile)
        self.assertIn("COPY overlay-dflash2/dflash2/", dockerfile)
        self.assertTrue((adapter / "upstream-Dockerfile.glm53-sm121-v9").is_file())
        self.assertTrue((adapter / "upstream-launch-tp4-24g.sh").is_file())
        operational = "\n".join((adapter / name).read_text(errors="ignore") for name in ("Dockerfile", "vllm-wrapper.py", "verify-runtime.py"))
        self.assertNotIn("ssh -", operational.lower())
        self.assertNotIn("nfs", operational.lower())

    def test_catalog_packages_bind_the_exact_current_recipes(self) -> None:
        index = load(ROOT / "catalog-index.json")
        entries = {Path(item["source_path"]).stem: item for item in index["recipes"]}
        for slug in RECIPES:
            if slug == "glm-5-3-flash-nvfp4-vllm-four":
                continue
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertEqual(entries[slug]["package"]["recipe_content_sha256"], digest(recipe))


if __name__ == "__main__": unittest.main()
