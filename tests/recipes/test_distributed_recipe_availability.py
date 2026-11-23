from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

from documents import recipe_display
from generated_catalog import GENERATED
from repo_root import ROOT

RECIPES = {
    "glm-5-2-quanttrio-vllm-four": 4,
    "glm-5-3-flash-nvfp4-vllm-four": 4,
    "inkling-small-nvfp4-sglang-dual": 2,
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


class DistributedRecipeAvailabilityTests(unittest.TestCase):
    def test_larger_topologies_and_candidate_metadata_are_explicit(self) -> None:
        for slug, nodes in RECIPES.items():
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            assert recipe["topology"]["node_count"] == nodes
            assert {"executable"} <= set(
                recipe_display(recipe["identity"]["slug"]).tags
            )
            assert recipe["models"]

    def test_per_node_disk_envelopes_are_bounded(self) -> None:
        for slug in RECIPES:
            disk = load(ROOT / "recipes" / f"{slug}.json")["topology"]["roles"][0][
                "resources"
            ]["disk"]
            assert disk["artifact_bytes"] > 0
            assert disk["working_bytes"] > 0

    def test_glm_tp4_tracks_current_public_source_profile(self) -> None:
        recipe = load(ROOT / "recipes/glm-5-3-flash-nvfp4-vllm-four.json")
        adapter = ROOT / "adapters/glm/tonyd2wild-glm53-tp4-current"
        dockerfile = (adapter / "Dockerfile").read_text()
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        assert (
            recipe["execution"]["build"]["context"]["path"]
            == "adapters/glm/tonyd2wild-glm53-tp4-current"
        )
        assert (
            recipe["provenance"]["source_reference"]
            == "https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/tree/2ac4e8dd661b12e6a1d7e69df711707abd6b9cd4"
        )
        assert recipe["execution"]["build"]["base_image"] == {
            "digest": "4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6",
            "repository": "ghcr.io/tonyd2wild/vllm-glm53-flash",
        }
        assert recipe["execution"]["build"]["network"]["hosts"] == []
        assert arguments["tensor-parallel-size"] == 4
        assert recipe["settings"]["context_tokens"]["value"] == 1048576
        assert arguments["max-model-len"] is None  # bound to context_tokens
        assert arguments["max-num-seqs"] == 64
        assert arguments["max-num-batched-tokens"] == 16384
        assert arguments["kv-cache-memory"] == 25769803776
        assert (
            arguments["speculative-config"]
            == '{"method":"dflash","model":"/models/drafter","num_speculative_tokens":7}'
        )
        assert not arguments["enforce-eager"]
        assert (
            arguments["compilation-config"] == '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
        )
        drafter = next(
            selection
            for selection in recipe["models"]
            if selection["id"] == "dependency-glm-5-3-flash-dflash2-bf582e4e"
        )
        assert {file["file_id"] for file in drafter["files"]} == {
            "metadata-70e0ed421d65-618cd5b83d",
            "readme-2014434d3e36-b335630551",
            "dflash2-figure-6d8dcc9a9472-e520c8f797",
            "config-c4aeac010119-587cb980af",
            "model-b038e1d9d1e7-9d75c1098f",
        }
        assert {file["mount"]["target"] for file in drafter["files"]} == {
            "/models/drafter"
        }
        assert 'ai.vonkforge.runtime-interface="v1"' in dockerfile
        # The kit's prebuilt image, by digest; the kit's patch stack is not rebuilt here.
        assert (
            "FROM ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef6"
            in dockerfile
        )
        assert "patch_v7" not in dockerfile
        assert "pip install" not in dockerfile
        assert (adapter / "sparse_attn_indexer_kpool_sm121.py").is_file()
        assert (adapter / "upstream-launch-tp4-24g.sh").is_file()
        operational = "\n".join(
            (adapter / name).read_text(errors="ignore")
            for name in ("Dockerfile", "vllm-wrapper.py")
        )
        assert "ssh -" not in operational.lower()
        assert "nfs" not in operational.lower()

    def test_catalog_packages_bind_the_exact_current_recipes(self) -> None:
        index = load(GENERATED / "catalog-index.json")
        entries = {Path(item["source_path"]).stem: item for item in index["recipes"]}
        for slug in RECIPES:
            if slug == "glm-5-3-flash-nvfp4-vllm-four":
                continue
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            assert entries[slug]["package"]["recipe_content_sha256"] == digest(recipe)


if __name__ == "__main__":
    unittest.main()
