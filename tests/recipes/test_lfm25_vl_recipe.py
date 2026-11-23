from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

from repo_root import ROOT

REVISION = "a3af5799199acdd2a4f56ac4342816abb46c12a9"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Lfm25VlRecipeTests(unittest.TestCase):
    def test_checkpoint_inventory_is_exact_and_complete(self) -> None:
        version = load(ROOT / "models/lfm2-5-vl-3b-bf16-a3af5799.json")
        assert version["source"]["revision"] == REVISION
        paths = {item["path"] for item in version["files"]}
        assert {
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
        } <= paths
        assert len(paths) == len(version["files"])

    def test_openai_contract_exposes_bounded_offline_multimodal_serving(self) -> None:
        recipe = load(ROOT / "recipes/lfm2-5-vl-3b-vllm-single.json")
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        assert arguments["limit-mm-per-prompt"] == '{"image":4}'
        assert arguments["allowed-local-media-path"] == "/inputs"
        assert recipe["interfaces"][0]["adapter"] == "openai"
        assert {item["name"] for item in recipe["runtime"]["environment"]} == {
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        }

    def test_recipe_pins_the_corrected_snapshot_and_source_bundles(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for slug, adapter in (
            ("lfm2-5-vl-3b-vllm-single", "adapters/liquidai/lfm25-vl-vllm"),
        ):
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            assert recipe["models"][0]["model"]["slug"] == "lfm2-5-vl-3b-bf16-a3af5799"
            archive, _, bundle_digest = tool["source_bundle"](ROOT / adapter)
            assert archive
            assert bundle_digest
            assert recipe["execution"]["build"]["context"]["path"] == adapter

    def test_adapter_uses_pinned_cuda13_arm64_image(self) -> None:
        source = (ROOT / "adapters/liquidai/lfm25-vl-vllm/Dockerfile").read_text()
        assert REVISION in source
        assert "@sha256:" in source


if __name__ == "__main__":
    unittest.main()
