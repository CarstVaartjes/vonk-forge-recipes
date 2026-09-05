from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LTX23_SLUG = "ltx-2-3-22b-distilled-1-1-diffusers-single"
FP4_SLUG = "ltx-2-19b-dev-fp4-pytorch-single"
BF16_SLUG = "ltx-2-19b-dev-bf16-diffusers-single"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_runtime_fixture(
    root: Path, files: list[tuple[str, str, str]]
) -> tuple[Path, Path]:
    """Create schema-2 selected files and their small mounted byte fixtures."""
    model_root = root / "models"
    document = load(ROOT / "tests/fixtures/compiled_workload_v2.json")
    artifacts = []
    for index, (mount_name, relative_path, file_id) in enumerate(files):
        payload = f"fixture-{index}".encode()
        materialized = model_root / mount_name / relative_path
        materialized.parent.mkdir(parents=True, exist_ok=True)
        materialized.write_bytes(payload)
        artifact = json.loads(json.dumps(document["artifacts"][index % 2]))
        digest = hashlib.sha256(payload).hexdigest()
        artifact.update(
            {
                "selection_id": "primary",
                "file_id": file_id,
                "path": relative_path,
                "sha256": digest,
                "size_bytes": len(payload),
                "mount": {"target": f"/models/{mount_name}", "read_only": True},
            }
        )
        artifact["model"].update(
            {
                "publisher": "lightricks",
                "slug": "ltx-fixture",
                "content_sha256": "a" * 64,
            }
        )
        artifact["distribution_object"].update(
            {
                "name": relative_path,
                "sha256": digest,
                "bytes": len(payload),
                "kind": "model",
            }
        )
        artifacts.append(artifact)
    document["artifacts"] = artifacts
    runtime_spec = root / "runtime.json"
    runtime_spec.write_text(json.dumps(document), encoding="utf-8")
    return model_root, runtime_spec


class LtxSparkFitTests(unittest.TestCase):
    def test_schema2_selected_file_materialization_checks_nested_path_and_size(
        self,
    ) -> None:
        namespace = runpy.run_path(
            str(ROOT / "adapters/video/ltx23-sync-native-disk/run.py")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, runtime_spec = canonical_runtime_fixture(
                root,
                [("target", "nested/ltx-2.3-22b-distilled-1.1.safetensors", "target")],
            )
            globals_ = namespace["_target_checkpoint"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            self.assertEqual(
                namespace["_target_checkpoint"](),
                model_root
                / "target"
                / "nested"
                / "ltx-2.3-22b-distilled-1.1.safetensors",
            )
            target = (
                model_root
                / "target"
                / "nested"
                / "ltx-2.3-22b-distilled-1.1.safetensors"
            )
            target.write_bytes(b"tampered")
            with self.assertRaisesRegex(SystemExit, "size changed"):
                namespace["_target_checkpoint"]()

    def test_schema2_allows_reused_content_across_selection_mounts(self) -> None:
        namespace = runpy.run_path(
            str(ROOT / "adapters/video/ltx23-sync-native-disk/run.py")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, runtime_spec = canonical_runtime_fixture(
                root,
                [
                    (
                        "target",
                        "nested/ltx-2.3-22b-distilled-1.1.safetensors",
                        "target",
                    ),
                    ("draft", "nested/ltx-2.3-22b-distilled-1.1.safetensors", "target"),
                ],
            )
            first = model_root / "target/nested/ltx-2.3-22b-distilled-1.1.safetensors"
            second = model_root / "draft/nested/ltx-2.3-22b-distilled-1.1.safetensors"
            second.write_bytes(first.read_bytes())
            document = load(runtime_spec)
            digest = hashlib.sha256(first.read_bytes()).hexdigest()
            document["artifacts"][1].update(
                {
                    "selection_id": "draft",
                    "sha256": digest,
                    "size_bytes": first.stat().st_size,
                }
            )
            document["artifacts"][1]["distribution_object"].update(
                {"sha256": digest, "bytes": first.stat().st_size}
            )
            runtime_spec.write_text(json.dumps(document), encoding="utf-8")
            globals_ = namespace["_target_checkpoint"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            artifacts = namespace["_runtime_artifacts"]()
            self.assertEqual(len(artifacts), 2)

    def test_schema2_rejects_legacy_artifact_size_shape(self) -> None:
        namespace = runpy.run_path(
            str(ROOT / "adapters/video/ltx23-sync-native-disk/run.py")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, runtime_spec = canonical_runtime_fixture(
                root,
                [("target", "ltx-2.3-22b-distilled-1.1.safetensors", "target")],
            )
            document = load(runtime_spec)
            artifact = document["artifacts"][0]
            artifact["bytes"] = artifact.pop("size_bytes")
            runtime_spec.write_text(json.dumps(document), encoding="utf-8")
            globals_ = namespace["_runtime_artifacts"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            with self.assertRaisesRegex(SystemExit, "artifact is invalid"):
                namespace["_runtime_artifacts"]()

    def test_pipeline_command_reuses_one_parsed_runtime_plan(self) -> None:
        namespace = runpy.run_path(
            str(ROOT / "adapters/video/ltx23-sync-native-disk/run.py")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, runtime_spec = canonical_runtime_fixture(
                root,
                [
                    (
                        "spatial-upscaler",
                        "nested/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                        "upscaler",
                    )
                ],
            )
            globals_ = namespace["_pipeline_command"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            target = model_root / "target/nested/ltx-2.3-22b-distilled-1.1.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"target-fixture")
            original_read_text = Path.read_text
            with mock.patch.object(
                namespace["Path"],
                "read_text",
                autospec=True,
                side_effect=lambda path, encoding=None: original_read_text(
                    path, encoding=encoding
                ),
            ) as read_text:
                namespace["_pipeline_command"](
                    target, root / "gemma", root / "output.mp4", 7, "prompt"
                )
            self.assertEqual(read_text.call_count, 1)

    def test_ltx2_dev_bf16_has_dedicated_disk_offload_and_97gb_admission(self) -> None:
        adapter_root = ROOT / "adapters/video/ltx23-sync-native-disk"
        namespace = runpy.run_path(str(adapter_root / "run.py"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, runtime_spec = canonical_runtime_fixture(
                root,
                [
                    (
                        "spatial-upscaler",
                        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
                        "spatial-upscaler",
                    ),
                    (
                        "distilled-lora",
                        "ltx-2-19b-distilled-lora-384.safetensors",
                        "distilled-lora",
                    ),
                ],
            )
            globals_ = namespace["_pipeline_command"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            target = model_root / "target" / "ltx-2-19b-dev.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"target-fixture")
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
            recipe["execution"]["build"]["context"]["path"],
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

        distilled = load(ROOT / "recipes/ltx-2-19b-distilled-diffusers-single.json")
        self.assertEqual(
            distilled["execution"]["build"]["context"]["path"],
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
            recipe["execution"]["build"]["context"]["path"],
            "adapters/video/ltx23-sync-native-disk",
        )

    def test_ltx23_dedicated_adapter_uses_disk_offload_and_is_bound(self) -> None:
        adapter_root = ROOT / "adapters/video/ltx23-sync-native-disk"
        namespace = runpy.run_path(str(adapter_root / "run.py"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, runtime_spec = canonical_runtime_fixture(
                root,
                [
                    (
                        "spatial-upscaler",
                        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                        "spatial-upscaler",
                    )
                ],
            )
            globals_ = namespace["_pipeline_command"].__globals__
            globals_["MODEL_ROOT"] = model_root
            globals_["RUNTIME_SPEC"] = runtime_spec
            target = model_root / "target" / "ltx-2.3-22b-distilled-1.1.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"target-fixture")
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
        self.assertEqual(
            recipe["execution"]["build"]["context"]["path"],
            "adapters/video/ltx23-sync-native-disk",
        )
        self.assertTrue(digest and archive)

    def test_fp4_snapshot_selects_exactly_the_runtime_inventory(self) -> None:
        recipe = load(ROOT / f"recipes/{FP4_SLUG}.json")
        model = load(ROOT / "models/ltx-2-19b-dev-fp4-dfcc2108.json")
        selected = recipe["models"][0]["files"]
        runtime_files = [
            item for item in model["files"] if item["roles"] != ["metadata"]
        ]
        expected_bytes = sum(item["size_bytes"] for item in runtime_files)
        self.assertEqual(
            {item["file_id"] for item in selected},
            {item["id"] for item in model["files"]},
        )
        self.assertGreater(expected_bytes, 0)

        disk = recipe["topology"]["roles"][0]["resources"]["disk"]
        self.assertGreaterEqual(disk["artifact_bytes"], expected_bytes)
        self.assertGreater(disk["staging_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
