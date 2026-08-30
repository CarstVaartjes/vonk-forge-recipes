from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIGEST = "15c98035c9bbba7ec61d25acd93c3c34b0516754c299813e5f51344e858abd2d"
RUNTIME_IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:"
    "41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
)


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def catalog_index_module():
    loader = importlib.machinery.SourceFileLoader(
        "vllm028_variant_catalog_index", str(ROOT / "tools/build-catalog-index")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Vllm028ModelVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gemma_fallback = load("recipes/gemma-4-26b-a4b-vllm-single.json")
        self.gemma = load("recipes/gemma-4-26b-a4b-vllm028-single.json")
        self.lfm_fallback = load("recipes/lfm2-5-vl-3b-vllm-single.json")
        self.lfm = load("recipes/lfm2-5-vl-3b-vllm028-single.json")

    def test_immutable_runtime_and_fallbacks_coexist(self) -> None:
        runtime = load("runtime-distributions/vllm-0-28-0-nvidia-arm64.json")
        self.assertEqual(runtime["image"], RUNTIME_IMAGE)
        self.assertEqual(digest(runtime), RUNTIME_DIGEST)
        self.assertEqual(
            self.gemma_fallback["runtime"]["distribution"]["slug"],
            "vllm-0-27-1-nvidia-arm64",
        )
        self.assertEqual(
            self.lfm_fallback["runtime"]["distribution"]["slug"],
            "vllm-0-27-1-cuda13-arm64",
        )
        for recipe in (self.gemma, self.lfm):
            with self.subTest(recipe=recipe["identity"]["slug"]):
                self.assertEqual(
                    recipe["runtime"]["distribution"],
                    {
                        "kind": "runtime-distribution",
                        "publisher": "vllm",
                        "slug": "vllm-0-28-0-nvidia-arm64",
                        "content_sha256": RUNTIME_DIGEST,
                    },
                )
                self.assertTrue(
                    {"executable", "candidate"} <= set(recipe["metadata"]["tags"])
                )
                self.assertIn("fallback", recipe["metadata"]["description"])

    def test_gemma_exact_image_chat_contract(self) -> None:
        arguments = {
            item["name"]: item["value"] for item in self.gemma["runtime"]["arguments"]
        }
        self.assertEqual(arguments["served-model-name"], "gemma-4-26b-a4b-it-vllm028")
        self.assertEqual(arguments["max-model-len"], 32_768)
        self.assertEqual(arguments["max-num-batched-tokens"], 4096)
        self.assertEqual(arguments["max-cudagraph-capture-size"], 2)
        self.assertEqual(json.loads(arguments["limit-mm-per-prompt"]), {"image": 4})
        self.assertEqual(arguments["allowed-local-media-path"], "/inputs")
        self.assertEqual(arguments["reasoning-parser"], "gemma4")
        self.assertNotIn("tool-call-parser", arguments)
        self.assertNotIn("enable-auto-tool-choice", arguments)
        self.assertTrue(
            {"multimodal", "vision", "image", "reasoning"}
            <= set(self.gemma["metadata"]["tags"])
        )
        self.assertEqual(
            self.gemma["interfaces"][0]["model_aliases"],
            ["gemma-4-26b-a4b-it-vllm028"],
        )
        self.assertIn(
            ("inputs", "/inputs", True),
            {
                (item["source"], item["target"], item["read_only"])
                for item in self.gemma["runtime"]["security"]["mounts"]
            },
        )

    def test_lfm_exact_image_and_tool_contract(self) -> None:
        arguments = {
            item["name"]: item["value"] for item in self.lfm["runtime"]["arguments"]
        }
        self.assertEqual(arguments["served-model-name"], "lfm2-5-vl-3b-vllm028")
        self.assertEqual(arguments["max-model-len"], 32_768)
        self.assertEqual(arguments["max-num-batched-tokens"], 4096)
        self.assertEqual(arguments["max-cudagraph-capture-size"], 4)
        self.assertEqual(json.loads(arguments["limit-mm-per-prompt"]), {"image": 4})
        self.assertEqual(arguments["allowed-local-media-path"], "/inputs")
        self.assertEqual(arguments["tool-call-parser"], "lfm2")
        self.assertIs(arguments["enable-auto-tool-choice"], True)
        self.assertTrue(
            {"multimodal", "vision-language", "image", "tool-use"}
            <= set(self.lfm["metadata"]["tags"])
        )
        self.assertEqual(
            self.lfm["interfaces"][0]["model_aliases"],
            ["lfm2-5-vl-3b-vllm028"],
        )
        self.assertIn(
            ("inputs", "/inputs", True),
            {
                (item["source"], item["target"], item["read_only"])
                for item in self.lfm["runtime"]["security"]["mounts"]
            },
        )

    def test_build_time_interface_screens_and_source_bundles_are_exact(self) -> None:
        module = catalog_index_module()
        cases = (
            (
                self.gemma,
                "adapters/google/gemma4-vllm-028",
                "Gemma4ForConditionalGeneration",
            ),
            (
                self.lfm,
                "adapters/liquidai/lfm25-vl-vllm-028",
                "Lfm2VLForConditionalGeneration",
            ),
        )
        for recipe, context_path, class_name in cases:
            with self.subTest(recipe=recipe["identity"]["slug"]):
                context = recipe["build"]["context"]
                archive, _, source_digest = module.source_bundle(ROOT / context_path)
                self.assertEqual(context["path"], context_path)
                self.assertEqual(context["expected_bytes"], len(archive))
                self.assertEqual(context["sha256"], source_digest)
                dockerfile = (ROOT / context_path / "Dockerfile").read_text()
                smoke = (ROOT / context_path / "model-interface-smoke.py").read_text()
                self.assertIn(RUNTIME_IMAGE, dockerfile)
                self.assertIn("python /tmp/model-interface-smoke.py", dockerfile)
                self.assertIn(class_name, smoke)
                self.assertIn("SupportsMultiModal", smoke)
                wrapper = ROOT / context_path / "vllm-wrapper.sh"
                self.assertTrue(wrapper.is_file())
                wrapper_text = wrapper.read_text()
                self.assertIn("expected the immutable model at /models", wrapper_text)
                self.assertIn("missing read-only multimodal input mount", wrapper_text)

    def test_releases_bind_the_exact_candidate_recipes(self) -> None:
        for slug, recipe in (
            ("gemma-4-26b-a4b-vllm028-single", self.gemma),
            ("lfm2-5-vl-3b-vllm028-single", self.lfm),
        ):
            with self.subTest(recipe=slug):
                release = load(f"recipe-releases/{slug}.json")
                self.assertEqual(release["version"], "1.0.0")
                self.assertEqual(release["released_at"], "2026-08-29")
                self.assertEqual(
                    release["history"][0]["recipe_content_sha256"], digest(recipe)
                )


if __name__ == "__main__":
    unittest.main()
