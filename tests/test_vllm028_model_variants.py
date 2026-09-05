from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import RecipeDefinition, content_sha256  # noqa: E402
RUNTIME_DIGEST = "15c98035c9bbba7ec61d25acd93c3c34b0516754c299813e5f51344e858abd2d"
RUNTIME_IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:"
    "41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
)


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(document: dict[str, object]) -> str:
    return content_sha256(RecipeDefinition.model_validate(document))


def catalog_entry(slug: str) -> dict[str, object]:
    catalog = load("catalog-index.json")
    return next(
        item for item in catalog["recipes"]
        if item["document"]["identity"]["slug"] == slug
    )


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
        for recipe in (self.gemma, self.lfm):
            self.assertEqual(recipe["execution"]["build"]["base_image"]["repository"], "docker.io/vllm/vllm-openai")
            self.assertEqual(recipe["execution"]["build"]["base_image"]["digest"], RUNTIME_IMAGE.split("@sha256:")[1])
        self.assertEqual(self.gemma_fallback["runtime"]["engine"], "vllm")
        self.assertEqual(self.lfm_fallback["runtime"]["engine"], "vllm")
        for recipe in (self.gemma, self.lfm):
            with self.subTest(recipe=recipe["identity"]["slug"]):
                self.assertEqual(
                    recipe["execution"]["build"]["base_image"]["digest"],
                    RUNTIME_IMAGE.split("@sha256:")[1],
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
        self.assertEqual(self.gemma["settings"]["context_tokens"]["value"], 32_768)
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
        self.assertEqual(self.gemma["interfaces"][0]["health_path"], "/v1/models")
        self.assertTrue(all(item["mount"]["read_only"] for item in self.gemma["models"][0]["files"]))

    def test_lfm_exact_image_and_tool_contract(self) -> None:
        arguments = {
            item["name"]: item["value"] for item in self.lfm["runtime"]["arguments"]
        }
        self.assertEqual(arguments["served-model-name"], "lfm2-5-vl-3b-vllm028")
        self.assertEqual(self.lfm["settings"]["context_tokens"]["value"], 32_768)
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
        self.assertEqual(self.lfm["interfaces"][0]["health_path"], "/v1/models")
        self.assertTrue(all(item["mount"]["read_only"] for item in self.lfm["models"][0]["files"]))

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
                context = recipe["execution"]["build"]["context"]
                archive, _, source_digest = module.source_bundle(ROOT / context_path)
                self.assertEqual(context["path"], context_path)
                self.assertEqual(context["path"], context_path)
                expected_digest = {"adapters/google/gemma4-vllm-028": "bcd5a8df070f142e831c74476df6cb639ecda313c6a38626d13e71500bd5ecc5", "adapters/liquidai/lfm25-vl-vllm-028": "8a70ef3c06595c0b32331113df0a34982bbcf38151b00e94725012cf013b177a"}[context_path]
                self.assertEqual(source_digest, expected_digest)
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

    def test_releases_and_packages_bind_exact_candidate_recipes(self) -> None:
        for slug, recipe, version, released_at in (
            ("gemma-4-26b-a4b-vllm028-single", self.gemma, "1.0.2", "2026-09-03"),
            ("lfm2-5-vl-3b-vllm028-single", self.lfm, "1.1.3", "2026-09-05"),
        ):
            with self.subTest(recipe=slug):
                definition = RecipeDefinition.model_validate(recipe)
                release = definition.release
                self.assertEqual(release.version, version)
                self.assertEqual(release.released_at, released_at)
                self.assertEqual(release.history[0].version, version)
                self.assertEqual(release.history[0].released_at, released_at)
                self.assertIsNone(release.history[0].prior_recipe_content_sha256)
                entry = catalog_entry(slug)
                recipe_digest = digest(recipe)
                self.assertEqual(
                    entry["content_sha256"], recipe_digest
                )
                package = entry["package"]
                self.assertEqual(package["recipe_content_sha256"], recipe_digest)
                payload = (ROOT / package["path"]).read_bytes()
                self.assertEqual(len(payload), package["expected_bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), package["sha256"])


if __name__ == "__main__":
    unittest.main()
