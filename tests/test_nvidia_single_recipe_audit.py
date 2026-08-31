from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPER_RECIPE = ROOT / "recipes/nemotron-3-super-120b-a12b-vllm-single.json"
SUPER_TARGET = ROOT / "model-versions/nemotron-3-super-120b-a12b-nvfp4.json"
SUPER_DRAFTER = (
    ROOT / "model-versions/nemotron-3-super-120b-a12b-bf16-mtpv2.json"
)
FLASH_RECIPE = ROOT / "recipes/nvidia-qwen-image-flash-diffusers-single.json"
FLASH_MODEL = ROOT / "model-versions/nvidia-qwen-image-flash.json"
FLASH_RELEASE = (
    ROOT / "recipe-releases/nvidia-qwen-image-flash-diffusers-single.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    payload = json.dumps(
        _read(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _arguments(recipe: dict[str, object]) -> dict[str, object]:
    return {
        str(item["name"]): item["value"]
        for item in recipe["runtime"]["arguments"]
    }


def _role_resources(recipe: dict[str, object]) -> dict[str, object]:
    return recipe["topology"]["roles"][0]["resources"]


class NvidiaSingleRecipeAuditTests(unittest.TestCase):
    def test_current_public_revisions_need_license_acceptance_but_no_token(self) -> None:
        expected_revisions = {
            SUPER_TARGET: "ff433f5493e25d631c9f12b5d55c674229923d02",
            SUPER_DRAFTER: "c929f8a55d0527fea9f58b4cedc9e0c855cfc421",
            FLASH_MODEL: "eafac15f6140e6dd9c6031217d658ac10bfb604b",
        }
        for path, revision in expected_revisions.items():
            with self.subTest(model_version=path.name):
                model = _read(path)
                self.assertEqual(model["source"]["revision"], revision)
                self.assertEqual(
                    model["access"],
                    {
                        "visibility": "public",
                        "gated": False,
                        "authentication": "none",
                    },
                )
                self.assertTrue(model["license"]["operator_acceptance_required"])

    def test_qwen_image_flash_inventory_closes_the_exact_snapshot(self) -> None:
        model = _read(FLASH_MODEL)
        revision = "eafac15f6140e6dd9c6031217d658ac10bfb604b"
        expected_paths = {
            ".gitattributes",
            "LICENSE",
            "README.md",
            "assets/qwen-image-flash-showcase.jpg",
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/added_tokens.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/model-00001-of-00001.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        }
        artifacts = model["artifacts"]
        self.assertEqual({item["path"] for item in artifacts}, expected_paths)
        self.assertEqual(len({item["id"] for item in artifacts}), 24)
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            model["sizes"]["download_bytes"],
        )
        self.assertEqual(model["sizes"]["download_bytes"], 57_708_363_017)
        self.assertEqual(model["parameters"]["total"], 28_850_000_000)
        self.assertEqual(model["limits"]["context_tokens"], 1024)
        self.assertEqual(model["limits"]["resolution_pixels"], 1024 * 1024)
        for artifact in artifacts:
            with self.subTest(artifact=artifact["path"]):
                self.assertEqual(artifact["revision"], revision)
                self.assertEqual(
                    artifact["repository"],
                    f"https://huggingface.co/nvidia/Qwen-Image-Flash/resolve/"
                    f"{revision}/{artifact['path']}",
                )
                self.assertRegex(artifact["sha256"], re.compile(r"^[0-9a-f]{64}$"))
                self.assertEqual(
                    artifact["download_bytes"], artifact["installed_bytes"]
                )

        large_artifacts = {item["path"]: item for item in artifacts}
        self.assertEqual(
            large_artifacts["transformer/model-00001-of-00001.safetensors"][
                "sha256"
            ],
            "3a716f870c1de89a9468629163c02fd80086b61d0216b6eb788854d72f223a3a",
        )
        self.assertEqual(
            large_artifacts["vae/diffusion_pytorch_model.safetensors"]["sha256"],
            "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",
        )

    def test_recipes_bind_current_models_and_supported_runtime_contracts(self) -> None:
        super_recipe = _read(SUPER_RECIPE)
        flash_recipe = _read(FLASH_RECIPE)
        self.assertEqual(
            super_recipe["model"]["content_sha256"],
            _canonical_digest(SUPER_TARGET),
        )
        self.assertEqual(
            super_recipe["dependencies"][0]["content_sha256"],
            _canonical_digest(SUPER_DRAFTER),
        )
        self.assertEqual(
            flash_recipe["model"]["content_sha256"],
            _canonical_digest(FLASH_MODEL),
        )
        self.assertEqual(
            super_recipe["runtime"]["distribution"]["slug"],
            "vllm-0-27-1-nvidia-arm64",
        )
        self.assertEqual(
            flash_recipe["runtime"]["distribution"]["slug"],
            "diffusers-0-40-0-cuda13-arm64",
        )

        super_arguments = _arguments(super_recipe)
        self.assertEqual(super_arguments["max-model-len"], 262_144)
        self.assertEqual(super_arguments["max-num-seqs"], 4)
        self.assertEqual(super_arguments["quantization"], "modelopt_fp4")
        self.assertEqual(super_arguments["moe-backend"], "marlin")
        self.assertEqual(super_arguments["kv-cache-dtype"], "fp8")
        self.assertEqual(super_arguments["mamba-ssm-cache-dtype"], "float16")
        self.assertEqual(
            json.loads(super_arguments["speculative-config"])["model"],
            "/models/drafter",
        )

        flash_arguments = _arguments(flash_recipe)
        self.assertEqual(
            {
                key: flash_arguments[key]
                for key in (
                    "num-inference-steps",
                    "true-cfg-scale",
                    "width",
                    "height",
                )
            },
            {
                "num-inference-steps": 4,
                "true-cfg-scale": "1",
                "width": 1024,
                "height": 1024,
            },
        )
        adapter = (
            ROOT
            / "adapters/image/nvidia-qwen-image-flash-diffusers/qwen_image_flash.py"
        ).read_text(encoding="utf-8")
        self.assertIn('from_pretrained(\n        "/models",', adapter)
        self.assertIn("dtype=torch.bfloat16", adapter)
        self.assertIn("local_files_only=True", adapter)
        self.assertIn("guidance_scale=None", adapter)
        self.assertIn("negative_prompt=None", adapter)

    def test_both_single_spark_contracts_fit_128gb_and_have_bounded_canaries(
        self,
    ) -> None:
        expected_disk = {
            SUPER_RECIPE: 286_751_720_953,
            FLASH_RECIPE: 195_125_089_051,
        }
        for path, expected_bytes in expected_disk.items():
            with self.subTest(recipe=path.name):
                recipe = _read(path)
                resources = _role_resources(recipe)
                disk = resources["disk"]
                memory = resources["memory"]
                self.assertEqual(sum(disk.values()), expected_bytes)
                startup_admission = (
                    memory["startup_peak_bytes"] + memory["system_reserve_bytes"]
                )
                steady_admission = (
                    memory["steady_state_bytes"]
                    + memory["runtime_growth_bytes"]
                    + memory["system_reserve_bytes"]
                )
                self.assertLessEqual(startup_admission, 128_000_000_000)
                self.assertLessEqual(steady_admission, 128_000_000_000)

        super_validation = _read(SUPER_RECIPE)["validation"]
        self.assertEqual(
            super_validation["validators"][0]["checks"],
            ["endpoint.healthy", "chat.nonempty", "chat.max-output-64"],
        )
        self.assertEqual(
            super_validation["benchmarks"][0]["configuration"],
            {"max_output_tokens": 64, "requests": 1, "timeout_seconds": 600},
        )

        flash_validation = _read(FLASH_RECIPE)["validation"]
        self.assertEqual(
            flash_validation["validators"][0]["checks"],
            ["artifact.mime.image-png"],
        )
        self.assertEqual(
            flash_validation["benchmarks"][0],
            {
                "name": "bounded-four-step-image",
                "framework": "diffusers",
                "configuration": {
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 4,
                    "timeout_seconds": 14400,
                },
            },
        )

    def test_qwen_flash_release_records_the_egress_fix(self) -> None:
        release = _read(FLASH_RELEASE)
        self.assertEqual(release["version"], "1.2.3")
        self.assertEqual(release["released_at"], "2026-09-01")
        self.assertEqual(release["history"][0]["version"], "1.2.3")
        self.assertEqual(release["history"][0]["upgrade_effect"], "rebuild")
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            _canonical_digest(FLASH_RECIPE),
        )


if __name__ == "__main__":
    unittest.main()
