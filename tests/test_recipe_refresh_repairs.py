from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts/src"))
from vonk_forge_contracts import RecipeDefinition, RecipeOptionError, document_sha256

HY3 = ROOT / "recipes/hy3-295b-nvfp4-mtp-tonyd2wild-vllm-dual.json"
LAGUNA = ROOT / "recipes/laguna-s-2-1-nvfp4-r0b0tlab-vllm-single.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class RecipeRefreshRepairTests(unittest.TestCase):
    def test_hy3_removed_choices_reject_stale_values_and_keep_visible_default(
        self,
    ) -> None:
        raw = load(HY3)
        recipe = RecipeDefinition.model_validate(raw)

        self.assertEqual(recipe.options, [])
        self.assertEqual(recipe.resolve_options(), {})
        self.assertEqual(recipe.with_option_choices({}).runtime, recipe.runtime)
        with self.assertRaises(RecipeOptionError):
            recipe.resolve_options({"speculative": "mtp2"})
        with self.assertRaises(RecipeOptionError):
            recipe.resolve_options({"execution": "graphs"})

        arguments = {item.name: item.value for item in recipe.runtime.arguments}
        speculative_config = arguments["speculative-config"]
        if not isinstance(speculative_config, str):
            self.fail("speculative-config is not a JSON string")
        self.assertEqual(
            json.loads(speculative_config),
            {"method": "mtp", "num_speculative_tokens": 1},
        )
        self.assertIs(arguments["enforce-eager"], True)

    def test_laguna_pair_follows_its_kit_and_matches_asset_envelope(self) -> None:
        recipe = load(LAGUNA)
        lock = load(ROOT / "adapters/llm/r0b0tlab-laguna-s-vllm-single/kit-lock.json")[
            "dependencies"
        ]
        expected_revisions = {
            "laguna-s-2-1-nvfp4-": lock["model"]["revision"],
            "laguna-s-2-1-dflash-nvfp4-": lock["drafter"]["revision"],
        }
        self.assertEqual(
            recipe["release"]["version"],
            "0.25.1-gb10-k7-"
            f"{lock['model']['revision'][:8]}-{lock['drafter']['revision'][:8]}",
        )
        self.assertEqual(
            recipe["execution"]["build"]["base_image"],
            {
                "digest": "4f8ba8a454fefc7b81f8e6ceafe46ed92ddf1f409a160556e5b1ea4daaaee80a",
                "repository": "ghcr.io/r0b0tlab/vllm-laguna-s-2.1-nvfp4-sm121",
            },
        )
        self.assertEqual(recipe["settings"]["context_tokens"]["value"], 262144)
        args = {
            item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]
        }
        self.assertEqual(
            args["speculative-config"],
            '{"method":"dflash","model":"/models/drafter","num_speculative_tokens":7}',
        )

        total_artifact_bytes = 0
        for selection in recipe["models"]:
            slug = selection["model"]["slug"]
            model = load(ROOT / "models" / f"{slug}.json")
            prefix = next(p for p in expected_revisions if slug.startswith(p))
            self.assertEqual(model["source"]["revision"], expected_revisions[prefix])
            self.assertEqual(
                document_sha256(model), selection["model"]["content_sha256"]
            )
            files_by_id = {item["id"]: item for item in model["files"]}
            selected_ids = {item["file_id"] for item in selection["files"]}
            self.assertEqual(selected_ids, set(files_by_id))
            mount_targets = {item["mount"]["target"] for item in selection["files"]}
            expected_mount = (
                "/models/target"
                if slug.startswith("laguna-s-2-1-nvfp4-")
                else "/models/drafter"
            )
            self.assertEqual(mount_targets, {expected_mount})
            total_artifact_bytes += sum(
                files_by_id[file_id]["size_bytes"] for file_id in selected_ids
            )

        declared = recipe["topology"]["roles"][0]["resources"]["disk"]["artifact_bytes"]
        self.assertEqual(declared, total_artifact_bytes)
        self.assertIn("vLLM 0.25.0 or later", recipe["metadata"]["description"])


if __name__ == "__main__":
    unittest.main()
