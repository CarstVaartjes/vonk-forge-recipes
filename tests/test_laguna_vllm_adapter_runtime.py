from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = (ROOT / "adapters/llm/laguna-vllm/Dockerfile", ROOT / "adapters/llm/laguna-s-vllm/Dockerfile")
RECIPE = ROOT / "recipes/laguna-xs-2-1-nvfp4-vllm-single.json"
S_RECIPE = ROOT / "recipes/laguna-s-2-1-nvfp4-vllm-single.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class LagunaVllmAdapterRuntimeTests(unittest.TestCase):
    def test_adapter_oci_labels_and_cache_contract_are_immutable(self) -> None:
        for path in DOCKERFILES:
            source = path.read_text(encoding="utf-8")
            self.assertIn("org.opencontainers.image.source=\"https://github.com/vllm-project/vllm\"", source)
            self.assertIn("org.opencontainers.image.licenses=\"Apache-2.0\"", source)
            self.assertIn("USER 10001:10001", source)
            self.assertIn("/outputs/cache", source)

    def test_model_selection_stays_outside_the_runtime_image(self) -> None:
        for path in (RECIPE, S_RECIPE):
            recipe = load(path)
            self.assertEqual(recipe["models"][0]["files"][0]["mount"]["target"], "/models")
            self.assertTrue(recipe["models"][0]["model"]["content_sha256"])
            self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_recipe_and_release_bind_the_source_bundle(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for path in (RECIPE, S_RECIPE):
            recipe = load(path)
            context = recipe["execution"]["build"]["context"]
            archive, _, bundle_digest = tool["source_bundle"](ROOT / context["path"])
            self.assertTrue(bundle_digest and archive)
            release = load(ROOT / "recipe-releases" / path.name)
            self.assertEqual(release["history"][0]["recipe_content_sha256"], digest(recipe))

    def test_runtime_is_offline_and_non_root(self) -> None:
        for path in (RECIPE, S_RECIPE):
            recipe = load(path)
            self.assertEqual({item["name"] for item in recipe["runtime"]["environment"]}, {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"})
            self.assertEqual(recipe["runtime"]["engine"], "vllm")


if __name__ == "__main__": unittest.main()
