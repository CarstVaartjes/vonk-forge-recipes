from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from generated_catalog import GENERATED

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/deepseek-v4-flash-vision-exp-6821d6ad.json"
RECIPE = ROOT / "recipes/deepseek-v4-flash-vision-exp-mia-dual.json"
ADAPTER = ROOT / "adapters/deepseek/mia-vllm-vision"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    body = json.dumps(load(path), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def digest_dict(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


class DeepSeekV4FlashVisionRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.recipe = map(load, (MODEL, RECIPE))

    def test_official_checkpoint_closure_is_exact(self) -> None:
        self.assertEqual(
            self.model["source"]["revision"], "6821d6ad3681a4b137b066b76094fa82ebd0a380"
        )
        self.assertGreater(len(self.model["files"]), 80)
        self.assertEqual(
            len({item["id"] for item in self.model["files"]}), len(self.model["files"])
        )
        self.assertTrue(all(item["sha256"] for item in self.model["files"]))

    def test_adapter_is_pinned_and_fail_closed(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        for value in (
            "97e8733238f81f5fdc44b241f8996a7858825744",
            'ai.vonkforge.runtime-interface="v1"',
            "getent passwd 10001",
        ):
            self.assertIn(value, dockerfile)

    def test_recipe_resolves_model_patch_and_dual_vision_profile(self) -> None:
        selected = self.recipe["models"][0]["model"]
        self.assertEqual(selected["content_sha256"], digest_dict(self.model))
        self.assertEqual(self.recipe["topology"]["node_count"], 2)
        self.assertEqual(self.recipe["topology"]["parallelism"]["tensor"], 2)
        self.assertEqual(self.recipe["topology"]["parallelism"]["backend"], "mp")
        arguments = {
            item["name"]: item["value"] for item in self.recipe["runtime"]["arguments"]
        }
        self.assertEqual(arguments["served-model-name"], "deepseek-v4-flash-vision-exp")
        self.assertEqual(self.recipe["settings"]["context_tokens"]["value"], 1_048_576)
        self.assertEqual(json.loads(arguments["limit-mm-per-prompt"]), {"image": 8})
        self.assertEqual(self.recipe["interfaces"][0]["adapter"], "openai")

    def test_options_default_to_the_upstream_defaults(self) -> None:
        from vonk_forge_contracts import read_recipe

        recipe = read_recipe(load(RECIPE))
        self.assertEqual(
            recipe.resolve_options(),
            {
                "thinking": "low",
                "prefill_lanes": "one",
                "draft_sampling": "probabilistic",
                "async_scheduling": "on",
                "reasoning_stops": "dormant",
                "scheduler_diagnostics": "off",
            },
        )
        base = {a.name: a.value for a in recipe.runtime.arguments}
        self.assertEqual(
            base["default-chat-template-kwargs"],
            '{"thinking":true,"reasoning_effort":"low"}',
        )
        self.assertIs(base["async-scheduling"], True)
        environment = {e.name: e.value for e in recipe.runtime.environment}
        self.assertEqual(environment["DSPARK_MAX_INFLIGHT_PREFILLS"], "1")
        # Choosing every default is the same runtime as choosing nothing.
        self.assertEqual(
            recipe.with_option_choices(recipe.resolve_options()).runtime,
            recipe.runtime,
        )

    def test_non_default_choices_change_only_their_binding(self) -> None:
        from vonk_forge_contracts import read_recipe

        recipe = read_recipe(load(RECIPE))
        chosen = recipe.with_option_choices(
            {
                "async_scheduling": "off",
                "prefill_lanes": "two",
                "draft_sampling": "greedy",
            }
        )
        arguments = {a.name: a.value for a in chosen.runtime.arguments}
        environment = {e.name: e.value for e in chosen.runtime.environment}
        self.assertIs(arguments["async-scheduling"], False)
        self.assertEqual(environment["DSPARK_MAX_INFLIGHT_PREFILLS"], "2")
        speculative = json.loads(str(arguments["speculative-config"]))
        self.assertEqual(speculative["draft_sample_method"], "greedy")

    def test_release_binds_the_current_recipe_digest(self) -> None:
        index = load(GENERATED / "catalog-index.json")
        entry = next(
            item
            for item in index["recipes"]
            if item["source_path"] == f"recipes/{RECIPE.name}"
        )
        self.assertEqual(entry["package"]["recipe_content_sha256"], digest(RECIPE))
        self.assertIn("candidate", self.recipe["metadata"]["tags"])


if __name__ == "__main__":
    unittest.main()
