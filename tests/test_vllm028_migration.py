from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_HASH = "15c98035c9bbba7ec61d25acd93c3c34b0516754c299813e5f51344e858abd2d"
IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:"
    "41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
)
MIGRATED = {
    "qwen3-5-9b-vllm-single": ("adapters/llm/qwen35-vllm", 8192, 4),
    "qwen3-6-27b-vllm-single": ("adapters/llm/vllm-openai-028", 8192, 4),
    "qwen3-8-27b-vllm-single": ("adapters/llm/vllm-openai-028", 8192, 4),
    "qwen3-8-27b-fp8-vllm-single": ("adapters/llm/vllm-openai-028", 8192, 8),
}


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def catalog_index_module():
    loader = importlib.machinery.SourceFileLoader(
        "vllm028_catalog_index", str(ROOT / "tools/build-catalog-index")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Vllm028MigrationTests(unittest.TestCase):
    def test_official_arm64_release_authority_is_immutable(self) -> None:
        runtime = load("runtime-distributions/vllm-0-28-0-nvidia-arm64.json")
        self.assertEqual(runtime["image"], IMAGE)
        self.assertEqual(
            runtime["source"]["revision"],
            "2cf0a6915ce544dc493a0990f2ea38d81601128a",
        )
        self.assertEqual(runtime["dependencies"][0]["version"], "0.28.0")
        self.assertEqual(runtime["image_manifest"]["compressed_layers_bytes"], 9_701_495_723)
        watch = load("upstream-watch.json")
        self.assertEqual(
            watch["overrides"][
                "runtime-distribution/vllm/vllm-0-28-0-nvidia-arm64"
            ]["policy"],
            "latest-release",
        )

    def test_migrated_recipes_pin_changed_vllm_defaults(self) -> None:
        module = catalog_index_module()
        for slug, (context_path, batched_tokens, graph_size) in MIGRATED.items():
            with self.subTest(recipe=slug):
                recipe = load(f"recipes/{slug}.json")
                runtime = recipe["runtime"]["distribution"]
                self.assertEqual(runtime["slug"], "vllm-0-28-0-nvidia-arm64")
                self.assertEqual(runtime["content_sha256"], RUNTIME_HASH)
                context = recipe["build"]["context"]
                self.assertEqual(context["path"], context_path)
                archive, _, digest = module.source_bundle(ROOT / context_path)
                self.assertEqual(context["sha256"], digest)
                self.assertEqual(context["expected_bytes"], len(archive))
                arguments = {
                    argument["name"]: argument["value"]
                    for argument in recipe["runtime"]["arguments"]
                }
                self.assertEqual(arguments["max-num-batched-tokens"], batched_tokens)
                self.assertEqual(arguments["max-cudagraph-capture-size"], graph_size)

    def test_gemma_is_plain_chat_until_bare_opener_is_fixed(self) -> None:
        # vLLM issue #53431: Gemma's canonical bare `:function` opener is silently
        # dropped in both 0.27.1 and 0.28.0, so the installable recipe must not
        # advertise a tool contract that its parser cannot honor.
        recipe = load("recipes/gemma-4-26b-a4b-vllm-single.json")
        self.assertNotIn("tool-use", recipe["metadata"]["tags"])
        arguments = {
            argument["name"]: argument["value"]
            for argument in recipe["runtime"]["arguments"]
        }
        self.assertNotIn("tool-call-parser", arguments)
        self.assertNotIn("enable-auto-tool-choice", arguments)
        self.assertEqual(arguments["reasoning-parser"], "gemma4")
        self.assertEqual(
            recipe["runtime"]["distribution"]["slug"],
            "vllm-0-27-1-nvidia-arm64",
        )

    def test_gemma_parser_smoke_covers_stream_and_nonstream_tool_json(self) -> None:
        smoke = (ROOT / "adapters/llm/vllm-openai-028/gemma4-parser-smoke.py").read_text(
            encoding="utf-8"
        )
        dockerfile = (ROOT / "adapters/llm/vllm-openai-028/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("def non_streaming()", smoke)
        self.assertIn("def streaming()", smoke)
        self.assertIn("call:set_status{active:true,count:42}", smoke)
        self.assertIn("json.loads(arguments)", smoke)
        self.assertIn("python /tmp/gemma4-parser-smoke.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
