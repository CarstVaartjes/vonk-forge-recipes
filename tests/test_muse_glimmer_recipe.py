from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from generated_catalog import GENERATED

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


class MuseGlimmerRecipeTests(unittest.TestCase):
    def test_exact_model_selection_and_file_closure(self) -> None:
        model = load(ROOT / "models/muse-glimmer-30b-bf16-a4e59da5.json")
        recipe = load(ROOT / "recipes/muse-glimmer-30b-bf16-vllm-single.json")
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertEqual(
            model["source"]["revision"], "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
        )
        self.assertTrue(
            model["files"] and all(item["sha256"] for item in model["files"])
        )

    def test_offline_single_spark_multimodal_contract(self) -> None:
        recipe = load(ROOT / "recipes/muse-glimmer-30b-bf16-vllm-single.json")
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        self.assertEqual(recipe["settings"]["context_tokens"]["value"], 32768)
        self.assertEqual(arguments["generation-config"], "auto")
        self.assertEqual(
            json.loads(arguments["limit-mm-per-prompt"]), {"image": 4, "video": 0}
        )
        self.assertEqual(recipe["interfaces"][0]["adapter"], "openai")
        self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_runtime_and_package_are_immutable(self) -> None:
        dockerfile = (ROOT / "adapters/llm/muse-glimmer-vllm/Dockerfile").read_text()
        self.assertIn("@sha256:", dockerfile)
        self.assertNotIn("huggingface.co", dockerfile)
        recipe = load(ROOT / "recipes/muse-glimmer-30b-bf16-vllm-single.json")
        index = load(GENERATED / "catalog-index.json")
        entry = next(
            item
            for item in index["recipes"]
            if item["source_path"] == f"recipes/{recipe['identity']['slug']}.json"
        )
        self.assertEqual(entry["package"]["recipe_content_sha256"], digest(recipe))

    def test_nvfp4_dflash_option_compiles_vllm_none_sentinel(self) -> None:
        from vonk_forge_contracts import read_recipe

        recipe = read_recipe(
            load(ROOT / "recipes/muse-glimmer-30b-nvfp4-vllm-mia-single.json")
        )
        default = recipe.with_option_choices({})
        default_value = next(
            item.value
            for item in default.runtime.arguments
            if item.name == "speculative-config"
        )
        self.assertEqual(
            json.loads(str(default_value)),
            {
                "method": "dflash",
                "model": "/models/drafter",
                "num_speculative_tokens": 16,
            },
        )
        default_kv = next(
            item.value
            for item in default.runtime.arguments
            if item.name == "kv-cache-dtype"
        )
        self.assertEqual(default_kv, "fp8_e4m3")

        auto_kv = recipe.with_option_choices({"kv-cache": "auto"})
        self.assertEqual(
            next(
                item.value
                for item in auto_kv.runtime.arguments
                if item.name == "kv-cache-dtype"
            ),
            "auto",
        )
        self.assertEqual(
            json.loads(
                str(
                    next(
                        item.value
                        for item in auto_kv.runtime.arguments
                        if item.name == "speculative-config"
                    )
                )
            ),
            {
                "method": "dflash",
                "model": "/models/drafter",
                "num_speculative_tokens": 16,
            },
        )

        disabled = recipe.with_option_choices({"speculative-decoding": "off"})
        disabled_value = next(
            item.value
            for item in disabled.runtime.arguments
            if item.name == "speculative-config"
        )
        self.assertEqual(disabled_value, "None")
        self.assertEqual(
            next(
                item.value
                for item in disabled.runtime.arguments
                if item.name == "kv-cache-dtype"
            ),
            "fp8_e4m3",
        )

        both = recipe.with_option_choices(
            {"speculative-decoding": "off", "kv-cache": "auto"}
        )
        args = {item.name: item.value for item in both.runtime.arguments}
        self.assertEqual(args["speculative-config"], "None")
        self.assertEqual(args["kv-cache-dtype"], "auto")


if __name__ == "__main__":
    unittest.main()
