from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LTX23_SLUG = "ltx-2-3-22b-distilled-1-1-diffusers-single"
FP4_SLUG = "ltx-2-19b-dev-fp4-pytorch-single"
BF16_SLUG = "ltx-2-19b-dev-bf16-diffusers-single"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class LtxSparkFitTests(unittest.TestCase):
    def test_ltx2_dev_bf16_has_dedicated_disk_offload_and_97gb_admission(self) -> None:
        adapter_root = ROOT / "adapters/video/ltx23-sync-native-disk"
        namespace = runpy.run_path(str(adapter_root / "run.py"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / "models"
            upscaler_root = model_root / "sha256" / ("a" * 64)
            upscaler_root.mkdir(parents=True)
            (upscaler_root / "ltx-2-spatial-upscaler-x2-1.0.safetensors").write_bytes(
                b"fixture"
            )
            lora_root = model_root / "sha256" / ("b" * 64)
            lora_root.mkdir(parents=True)
            (lora_root / "ltx-2-19b-distilled-lora-384.safetensors").write_bytes(
                b"fixture"
            )
            runtime_spec = root / "runtime.json"
            runtime_spec.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "kind": "http.file",
                                "repository": (
                                    "https://huggingface.co/Lightricks/LTX-2/resolve/"
                                    "dfcc2108383fe1aaa0584bdf55d368a4bdadd90c/"
                                    "ltx-2-spatial-upscaler-x2-1.0.safetensors"
                                ),
                                "revision": "sha256:" + ("a" * 64),
                                "path": str(upscaler_root),
                            },
                            {
                                "kind": "http.file",
                                "repository": (
                                    "https://huggingface.co/Lightricks/LTX-2/resolve/"
                                    "dfcc2108383fe1aaa0584bdf55d368a4bdadd90c/"
                                    "ltx-2-19b-distilled-lora-384.safetensors"
                                ),
                                "revision": "sha256:" + ("b" * 64),
                                "path": str(lora_root),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            globals_ = namespace["_pipeline_command"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            target = root / "ltx-2-19b-dev.safetensors"
            target.write_bytes(b"fixture")
            command = namespace["_pipeline_command"](
                target,
                root / "gemma",
                root / "output.mp4",
                7,
                "Operator supplied prompt",
            )

        self.assertEqual(command[command.index("--offload") + 1], "disk")
        self.assertEqual(command[2], "ltx_pipelines.ti2vid_two_stages")
        recipe = load(ROOT / f"recipes/{BF16_SLUG}.json")
        self.assertEqual(
            recipe["build"]["context"]["path"],
            "adapters/video/ltx23-sync-native-disk",
        )
        self.assertIn("disk-offload", recipe["metadata"]["tags"])
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(
            (
                memory["startup_peak_bytes"],
                memory["steady_state_bytes"],
                memory["runtime_growth_bytes"],
                memory["system_reserve_bytes"],
            ),
            (89_000_000_000, 75_000_000_000, 8_000_000_000, 8_000_000_000),
        )
        self.assertEqual(
            max(
                memory["startup_peak_bytes"],
                memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
            )
            + memory["system_reserve_bytes"],
            97_000_000_000,
        )

        distilled = load(
            ROOT / "recipes/ltx-2-19b-distilled-diffusers-single.json"
        )
        self.assertEqual(
            distilled["build"]["context"]["path"],
            "adapters/video/ltx2-sync-native",
        )

    def test_ltx23_disk_offload_candidate_is_admissible_but_unaccepted(self) -> None:
        recipe = load(ROOT / f"recipes/{LTX23_SLUG}.json")
        tags = set(recipe["metadata"]["tags"])
        self.assertTrue({"executable", "candidate", "disk-offload"} <= tags)
        self.assertNotIn("accepted", tags)
        self.assertNotIn("hardware-blocked", tags)

        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(memory["startup_peak_bytes"], 93_000_000_000)
        self.assertEqual(memory["steady_state_bytes"], 77_000_000_000)
        self.assertEqual(memory["runtime_growth_bytes"], 8_000_000_000)
        self.assertEqual(memory["system_reserve_bytes"], 8_000_000_000)
        workload_peak = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        )
        self.assertEqual(
            workload_peak + memory["system_reserve_bytes"], 101_000_000_000
        )
        self.assertLessEqual(
            workload_peak + memory["system_reserve_bytes"], 128_000_000_000
        )
        for fact in ("93 GB", "77 GB", "31 GB", "8 GB"):
            self.assertIn(fact, recipe["metadata"]["description"])
        self.assertIn(
            "Physical Spark acceptance remains pending",
            recipe["metadata"]["description"],
        )
        self.assertEqual(
            recipe["build"]["context"]["path"],
            "adapters/video/ltx23-sync-native-disk",
        )

        for ledger_name in ("audio", "video"):
            ledger = load(ROOT / f"model-targets/{ledger_name}.json")
            target = next(
                item
                for item in ledger["targets"]
                if item.get("catalog_model_version") == "ltx-2-3-22b-distilled-1-1"
            )
            self.assertEqual(target["status"], "candidate")
            self.assertEqual(target["recipe_slugs"], [LTX23_SLUG])
            for fact in ("93 GB", "77 GB", "31 GB", "8 GB", "128 GB"):
                self.assertIn(fact, target["notes"])
            self.assertIn("lowest-memory disk offload", target["notes"])
            self.assertIn("Physical Spark acceptance remains pending", target["notes"])

    def test_ltx23_dedicated_adapter_uses_disk_offload_and_is_bound(self) -> None:
        adapter_root = ROOT / "adapters/video/ltx23-sync-native-disk"
        namespace = runpy.run_path(str(adapter_root / "run.py"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / "models"
            artifact_root = model_root / "sha256" / ("a" * 64)
            artifact_root.mkdir(parents=True)
            upscaler = artifact_root / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
            upscaler.write_bytes(b"fixture")
            runtime_spec = root / "runtime.json"
            runtime_spec.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "kind": "http.file",
                                "repository": (
                                    "https://huggingface.co/Lightricks/LTX-2.3/resolve/"
                                    "6b5a83e3045eaf8e46cfa0acce512412aa2b9cce/"
                                    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
                                ),
                                "revision": "sha256:" + ("b" * 64),
                                "path": str(artifact_root),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            globals_ = namespace["_pipeline_command"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            target = root / "ltx-2.3-22b-distilled-1.1.safetensors"
            target.write_bytes(b"fixture")
            command = namespace["_pipeline_command"](
                target,
                root / "gemma",
                root / "output.mp4",
                7,
                "Operator supplied prompt",
            )

        self.assertEqual(command[command.index("--offload") + 1], "disk")
        self.assertEqual(command[2], "ltx_pipelines.distilled")
        self.assertEqual(
            Path(command[command.index("--spatial-upsampler-path") + 1]).name,
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        )

        recipe = load(ROOT / f"recipes/{LTX23_SLUG}.json")
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(adapter_root)
        self.assertEqual(recipe["build"]["context"]["sha256"], digest)
        self.assertEqual(recipe["build"]["context"]["expected_bytes"], len(archive))

    def test_fp4_snapshot_selects_exactly_the_runtime_inventory(self) -> None:
        recipe = load(ROOT / f"recipes/{FP4_SLUG}.json")
        model = load(ROOT / "model-versions/ltx-2-19b-dev-fp4-dfcc2108.json")
        artifact = recipe["artifacts"][0]
        runtime_files = [
            item for item in model["artifacts"] if item["roles"] != ["metadata"]
        ]
        expected_paths = sorted(item["path"] for item in runtime_files)
        expected_bytes = sum(item["download_bytes"] for item in runtime_files)

        self.assertEqual(artifact["kind"], "huggingface.snapshot")
        self.assertEqual(artifact["include_paths"], expected_paths)
        self.assertEqual(len(expected_paths), 25)
        self.assertEqual(artifact["download_bytes"], expected_bytes)
        self.assertEqual(artifact["installed_bytes"], expected_bytes)
        self.assertNotIn("README.md", artifact["include_paths"])
        self.assertNotIn("LICENSE", artifact["include_paths"])
        self.assertLess(expected_bytes, model["sizes"]["download_bytes"])

        build = recipe["build"]["resources"]
        disk = recipe["topology"]["roles"][0]["resources"]["disk"]
        peer = load(ROOT / "recipes/ltx-2-19b-dev-bf16-diffusers-single.json")
        peer_image_bytes = peer["topology"]["roles"][0]["resources"]["disk"][
            "image_bytes"
        ]
        self.assertEqual(disk["image_bytes"], peer_image_bytes)
        self.assertEqual(build["download_bytes"], expected_bytes + peer_image_bytes)
        self.assertEqual(build["temporary_bytes"], expected_bytes * 2)
        self.assertEqual(disk["artifact_bytes"], expected_bytes)
        self.assertEqual(disk["staging_bytes"], expected_bytes * 2)


if __name__ == "__main__":
    unittest.main()
