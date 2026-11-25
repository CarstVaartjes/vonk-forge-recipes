from __future__ import annotations

import json
import re
import unittest
from typing import Any

from documents import recipe_display
from repo_root import ROOT

from recipe_tooling import catalog_index

IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:"
    "e06f024cf1dd864a2f4b6e033b496807c55952404c14cf8c1b08e4ca9888597e"
)
MIGRATED = {
    "qwen3-5-9b-vllm-single": ("adapters/llm/qwen35-vllm", 8192, 4),
    "qwen3-6-27b-vllm-single": ("adapters/llm/vllm-openai-028", 8192, 4),
    "qwen3-8-27b-vllm-single": ("adapters/llm/vllm-openai-028", 8192, 4),
    "qwen3-8-27b-fp8-vllm-single": ("adapters/llm/vllm-openai-028", 8192, 8),
}


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def catalog_index_module():
    return catalog_index


class Vllm028MigrationTests(unittest.TestCase):
    def test_official_arm64_runtime_authority_is_immutable(self) -> None:
        recipe = load("recipes/qwen3-6-27b-vllm-single.json")
        base = recipe["execution"]["build"]["base_image"]
        assert f"{base['repository']}@sha256:{base['digest']}" == IMAGE
        dockerfile = (ROOT / recipe["execution"]["build"]["dockerfile"]).read_text()
        assert "ced6857afa0ea7b2e3f0846a62e1394e90f15607" in dockerfile
        assert "vllm-openai" in base["repository"]

    def test_migrated_recipes_pin_changed_vllm_defaults(self) -> None:
        module = catalog_index_module()
        for slug, (context_path, batched_tokens, graph_size) in MIGRATED.items():
            with self.subTest(recipe=slug):
                recipe = load(f"recipes/{slug}.json")
                assert recipe["runtime"]["engine"] == "vllm"
                context = recipe["execution"]["build"]["context"]
                assert context["path"] == context_path
                _archive, _, digest = module.source_bundle(ROOT / context_path)
                assert re.search(r"^[a-f0-9]{64}$", digest)
                arguments = {
                    argument["name"]: argument["value"]
                    for argument in recipe["runtime"]["arguments"]
                }
                assert arguments["max-num-batched-tokens"] == batched_tokens
                assert arguments["max-cudagraph-capture-size"] == graph_size

    def test_gemma_tool_contract_requires_the_bare_opener_fix(self) -> None:
        # vLLM issue #53431: Gemma's canonical bare `:function` opener was
        # silently dropped in 0.27.1 and 0.28.0. The fix (#53444) ships in
        # 0.30.0, the first runtime allowed to advertise the tool contract.
        recipe = load("recipes/gemma-4-26b-a4b-vllm-single.json")
        dockerfile = (ROOT / "adapters/llm/vllm-openai/Dockerfile").read_text()
        assert "ced6857afa0ea7b2e3f0846a62e1394e90f15607" in dockerfile
        assert "tool-use" in recipe_display(recipe["identity"]["slug"]).tags
        arguments = {
            argument["name"]: argument["value"]
            for argument in recipe["runtime"]["arguments"]
        }
        assert arguments["tool-call-parser"] == "gemma4"
        assert arguments["enable-auto-tool-choice"] is True
        assert arguments["reasoning-parser"] == "gemma4"
        assert recipe["runtime"]["engine"] == "vllm"


if __name__ == "__main__":
    unittest.main()
