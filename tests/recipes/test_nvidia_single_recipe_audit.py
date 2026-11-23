from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

from repo_root import ROOT

from vonk_forge_contracts import document_sha256

SUPER_RECIPE = ROOT / "recipes/nemotron-3-super-120b-a12b-vllm-single.json"
FLASH_RECIPE = ROOT / "recipes/nvidia-qwen-image-flash-diffusers-single.json"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(document: dict[str, Any]) -> str:
    return document_sha256(document)


def model_for(recipe: dict[str, Any], index: int = 0) -> dict[str, Any]:
    slug = recipe["models"][index]["model"]["slug"]
    return read(ROOT / "models" / f"{slug}.json")


def args(recipe: dict[str, Any]) -> dict[str, Any]:
    return {item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]}


class NvidiaSingleRecipeAuditTests(unittest.TestCase):
    def test_current_model_snapshots_are_pinned(self) -> None:
        for recipe_path in (SUPER_RECIPE, FLASH_RECIPE):
            recipe = read(recipe_path)
            with self.subTest(recipe=recipe_path.name):
                model = model_for(recipe)
                assert re.search(r"^[a-f0-9]{40,64}$", model["source"]["revision"])

    def test_qwen_flash_snapshot_and_job_contract(self) -> None:
        recipe = read(FLASH_RECIPE)
        model = model_for(recipe)
        assert model["source"]["revision"] == "eafac15f6140e6dd9c6031217d658ac10bfb604b"
        assert len(model["files"]) == 24
        assert recipe["models"][0]["model"]["content_sha256"] == digest(model)
        assert recipe["interfaces"][0]["adapter"] == "image-job"
        assert {
            item["name"]: item.get("value") for item in recipe["runtime"]["arguments"]
        } == {
            "output-mime": "image/png",
            "pipeline": "text-to-image",
            "seed": 42,
            "num-inference-steps": 4,
            "true-cfg-scale": "1",
            "width": 1024,
            "height": 1024,
        }

    def test_super_binds_target_and_drafter_files_with_mtp_runtime(self) -> None:
        recipe = read(SUPER_RECIPE)
        target = model_for(recipe)
        drafter = model_for(recipe, 1)
        assert (
            target["source"]["revision"] == "ff433f5493e25d631c9f12b5d55c674229923d02"
        )
        assert (
            drafter["source"]["revision"] == "c929f8a55d0527fea9f58b4cedc9e0c855cfc421"
        )
        assert len(recipe["models"]) == 2
        assert args(recipe)["quantization"] == "modelopt_fp4"
        assert args(recipe)["moe-backend"] == "marlin"
        assert (
            json.loads(args(recipe)["speculative-config"])["model"] == "/models/drafter"
        )

    def test_single_spark_memory_admission_and_representative_checks(self) -> None:
        for path in (SUPER_RECIPE, FLASH_RECIPE):
            recipe: dict[str, Any] = read(path)
            assert len(recipe["validation"]["serving"]["checks"]) >= 1


if __name__ == "__main__":
    unittest.main()
