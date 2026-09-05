from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import ModelDefinition, content_sha256  # noqa: E402
SUPER_RECIPE = ROOT / "recipes/nemotron-3-super-120b-a12b-vllm-single.json"
FLASH_RECIPE = ROOT / "recipes/nvidia-qwen-image-flash-diffusers-single.json"


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict[str, object]) -> str:
    return content_sha256(ModelDefinition.model_validate(document))


def model_for(recipe: dict[str, object], index: int = 0) -> dict[str, object]:
    slug = recipe["models"][index]["model"]["slug"]  # type: ignore[index]
    return read(ROOT / "models" / f"{slug}.json")


def args(recipe: dict[str, object]) -> dict[str, object]:
    return {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}  # type: ignore[index]


class NvidiaSingleRecipeAuditTests(unittest.TestCase):
    def test_current_model_snapshots_require_operator_acceptance(self) -> None:
        for recipe_path in (SUPER_RECIPE, FLASH_RECIPE):
            recipe = read(recipe_path)
            with self.subTest(recipe=recipe_path.name):
                model = model_for(recipe)
                self.assertRegex(model["source"]["revision"], r"^[a-f0-9]{40,64}$")
                self.assertTrue(model["license"]["operator_acceptance_required"])
                self.assertNotIn("token", json.dumps(model["license"]).lower())

    def test_qwen_flash_snapshot_and_job_contract(self) -> None:
        recipe = read(FLASH_RECIPE)
        model = model_for(recipe)
        self.assertEqual(model["source"]["revision"], "eafac15f6140e6dd9c6031217d658ac10bfb604b")
        self.assertEqual(len(model["files"]), 24)
        self.assertEqual(model["parameters"]["total"], 28_850_000_000)
        self.assertEqual(model["limits"]["resolution_pixels"], 1024 * 1024)
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], digest(model))
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertEqual({item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}, {"output-mime": "image/png", "pipeline": "text-to-image", "seed": 42, "num-inference-steps": 4, "true-cfg-scale": "1", "width": 1024, "height": 1024})

    def test_super_binds_target_and_drafter_files_with_mtp_runtime(self) -> None:
        recipe = read(SUPER_RECIPE)
        target = model_for(recipe)
        drafter = model_for(recipe, 1)
        self.assertEqual(target["source"]["revision"], "ff433f5493e25d631c9f12b5d55c674229923d02")
        self.assertEqual(drafter["source"]["revision"], "c929f8a55d0527fea9f58b4cedc9e0c855cfc421")
        self.assertEqual(len(recipe["models"]), 2)
        self.assertEqual(args(recipe)["quantization"], "modelopt_fp4")
        self.assertEqual(args(recipe)["moe-backend"], "marlin")
        self.assertEqual(json.loads(args(recipe)["speculative-config"])["model"], "/models/drafter")
        self.assertTrue(model_for(recipe)["license"]["operator_acceptance_required"])

    def test_single_spark_memory_admission_and_representative_checks(self) -> None:
        for path in (SUPER_RECIPE, FLASH_RECIPE):
            recipe = read(path)
            resources = recipe["topology"]["roles"][0]["resources"]  # type: ignore[index]
            memory = resources["memory"]
            self.assertLessEqual(memory["startup_peak_bytes"] + memory["system_reserve_bytes"], 128_000_000_000)
            self.assertLessEqual(memory["steady_state_bytes"] + memory["runtime_growth_bytes"] + memory["system_reserve_bytes"], 128_000_000_000)
            self.assertGreaterEqual(len(recipe["validation"]["serving"]["checks"]), 1)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
