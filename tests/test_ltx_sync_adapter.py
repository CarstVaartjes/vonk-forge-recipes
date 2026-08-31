# ruff: noqa: S102 -- isolated synthetic adapter modules are executed in tests.
from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/video/ltx2-sync-native"
ADAPTER_PATH = ADAPTER_ROOT / "run.py"
RUNTIME_PATH = ROOT / "runtime-distributions/ltx2-pipelines-1-3-arm64.json"
GEMMA_PATH = ROOT / "model-versions/ltx-2-gemma3-text-encoder-dfcc2108.json"
SOURCE_REVISION = "a95ab856bf29407b6b066ede0abe1846050db56c"
SOURCE_ARCHIVE_SHA256 = (
    "4698fc5f635196edc08e891f209402d6b80e0b64d6c55589266e2448966500e8"
)
MODEL_SPECS = {
    "ltx-2-19b-distilled-diffusers-single": "ltx-2-19b-distilled.json",
    "ltx-2-19b-distilled-fp8-diffusers-single": "ltx-2-19b-distilled-fp8.json",
}


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    payload = json.dumps(
        _document(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _adapter_module():
    module = types.ModuleType("ltx_sync_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(
        compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"),
        module.__dict__,
    )
    return module


class LtxSyncAuthorityTests(unittest.TestCase):
    def test_recipes_resolve_exact_runnable_authorities_and_closure(self) -> None:
        runtime_digest = _canonical_digest(RUNTIME_PATH)
        gemma_digest = _canonical_digest(GEMMA_PATH)
        gemma = _document(GEMMA_PATH)
        gemma_files = {
            artifact["repository"]: artifact for artifact in gemma["artifacts"]
        }

        for slug, model_name in MODEL_SPECS.items():
            recipe = _document(ROOT / "recipes" / f"{slug}.json")
            model_path = ROOT / "model-versions" / model_name
            model = _document(model_path)
            primary_files = {
                artifact["repository"]: artifact for artifact in model["artifacts"]
            }

            self.assertEqual(
                recipe["model"]["content_sha256"], _canonical_digest(model_path)
            )
            self.assertEqual(
                recipe["runtime"]["distribution"]["content_sha256"], runtime_digest
            )
            self.assertEqual(recipe["dependencies"][0]["content_sha256"], gemma_digest)
            self.assertEqual(model["dependencies"], recipe["dependencies"])
            self.assertIsNone(recipe["execution"]["patch_bundle"])
            prompt_input = recipe["interfaces"][0]["input"]
            self.assertTrue(prompt_input["required"])
            self.assertEqual(prompt_input["max_bytes"], 16 * 1024)
            self.assertEqual(prompt_input["slots"][0]["id"], "prompt")
            self.assertEqual(prompt_input["slots"][0]["min_files"], 1)
            self.assertIn(
                {"source": "inputs", "target": "/inputs", "read_only": True},
                recipe["runtime"]["security"]["mounts"],
            )
            output_contract = recipe["interfaces"][0]["output"]
            self.assertEqual(output_contract["path"], "/outputs")
            self.assertEqual(output_contract["max_total_bytes"], 1024**3)
            self.assertEqual(output_contract["slots"][0]["media_types"], ["video/mp4"])
            self.assertEqual(output_contract["slots"][0]["min_files"], 1)
            self.assertEqual(output_contract["slots"][0]["max_files"], 1)

            tags = set(recipe["metadata"]["tags"])
            self.assertTrue({"candidate", "video", "audio", "synchronized"} <= tags)
            self.assertTrue(
                {"metadata-only", "non-executable", "integration-required"}.isdisjoint(
                    tags
                )
            )

            expected_files = primary_files | gemma_files
            self.assertEqual(len(recipe["artifacts"]), len(expected_files))
            self.assertEqual(recipe["artifacts"][0]["id"], "target")
            mount_targets = set()
            for artifact in recipe["artifacts"]:
                authority = expected_files[artifact["repository"]]
                self.assertEqual(artifact["revision"], f"sha256:{authority['sha256']}")
                self.assertEqual(
                    artifact["download_bytes"], authority["download_bytes"]
                )
                expected_mount = f"/models/{artifact['id']}"
                self.assertEqual(artifact["mount"]["target"], expected_mount)
                mount_targets.add(expected_mount)
            self.assertEqual(len(mount_targets), len(recipe["artifacts"]))
            self.assertIn("/models/target", mount_targets)

    def test_signed_source_bundle_matches_every_recipe(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        for slug in MODEL_SPECS:
            context = _document(ROOT / "recipes" / f"{slug}.json")["build"]["context"]
            self.assertEqual(context["sha256"], digest)
            self.assertEqual(context["expected_bytes"], len(archive))

    def test_container_is_pinned_and_runtime_is_offline(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        runner = ADAPTER_PATH.read_text(encoding="utf-8")
        runtime = _document(RUNTIME_PATH)

        self.assertEqual(runtime["source"]["revision"], SOURCE_REVISION)
        self.assertEqual(
            runtime["source"]["archive_sha256"], SOURCE_ARCHIVE_SHA256
        )
        self.assertIn(f"LTX-2/archive/{SOURCE_REVISION}.tar.gz", dockerfile)
        self.assertIn(SOURCE_ARCHIVE_SHA256, dockerfile)
        self.assertIn("sha256sum --check --strict", dockerfile)
        self.assertIn("torchvision==0.28.0", dockerfile)
        self.assertIn(
            "nvidia-cudnn-cu13==9.24.0.43",
            (ADAPTER_ROOT / "requirements.lock").read_text(encoding="utf-8"),
        )
        self.assertTrue(runtime["build"]["offline_after_installation"])
        self.assertEqual(runtime["security"]["network_mode"], "none")
        self.assertNotIn("huggingface.co", runner)
        self.assertNotIn("requests", runner)
        self.assertIn("VONK_RUNTIME_SPEC", runner)


class LtxSyncRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _adapter_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.module.MODEL_ROOT = self.root / "models"
        self.module.MODEL_ROOT.mkdir()
        self.module.RUNTIME_SPEC = self.root / "runtime.json"
        self.module.INPUT_ROOT = self.root / "inputs"
        self.module.INPUT_ROOT.mkdir()
        self.prompt_path = self.module.INPUT_ROOT / "scene.txt"

        files = {
            **self.module.GEMMA_FILES,
            "ltx-2-spatial-upscaler-x2-1.0.safetensors": (
                "ltx-2-spatial-upscaler-x2-1.0.safetensors"
            ),
            "ltx-2-19b-distilled-lora-384.safetensors": (
                "ltx-2-19b-distilled-lora-384.safetensors"
            ),
        }
        artifacts = []
        for index, (repository_suffix, filename) in enumerate(files.items(), start=1):
            mount = self.module.MODEL_ROOT / "sha256" / f"{index:064x}"
            mount.mkdir(parents=True)
            (mount / filename).write_bytes(b"fixture")
            (mount / ".vonk-manifest.json").write_text("{}", encoding="utf-8")
            artifacts.append(
                {
                    "kind": "http.file",
                    "repository": f"https://models.example/{repository_suffix}",
                    "revision": f"sha256:{index:064x}",
                    "path": str(mount),
                }
            )
        self.module.RUNTIME_SPEC.write_text(
            json.dumps({"artifacts": artifacts}), encoding="utf-8"
        )

    def _target(self, filename: str) -> Path:
        mount = self.module.MODEL_ROOT / "sha256" / f"{999:064x}"
        mount.mkdir(exist_ok=True)
        for path in mount.iterdir():
            path.unlink()
        target = mount / filename
        target.write_bytes(b"fixture")
        (mount / ".vonk-manifest.json").write_text("{}", encoding="utf-8")
        document = json.loads(self.module.RUNTIME_SPEC.read_text(encoding="utf-8"))
        document["artifacts"] = [
            artifact
            for artifact in document["artifacts"]
            if not any(
                artifact["repository"].endswith(f"/{candidate}")
                for candidate in self.module.TARGET_FILENAMES
            )
        ]
        document["artifacts"].append(
            {
                "kind": "http.file",
                "repository": f"https://models.example/{filename}",
                "revision": f"sha256:{999:064x}",
                "path": str(mount),
            }
        )
        self.module.RUNTIME_SPEC.write_text(json.dumps(document), encoding="utf-8")
        return target

    def _upscaler(self, filename: str) -> Path:
        mount = self.module.MODEL_ROOT / "sha256" / f"{998:064x}"
        mount.mkdir(exist_ok=True)
        for path in mount.iterdir():
            path.unlink()
        upscaler = mount / filename
        upscaler.write_bytes(b"fixture")
        (mount / ".vonk-manifest.json").write_text("{}", encoding="utf-8")
        document = json.loads(self.module.RUNTIME_SPEC.read_text(encoding="utf-8"))
        document["artifacts"] = [
            artifact
            for artifact in document["artifacts"]
            if "spatial-upscaler" not in artifact["repository"]
        ]
        document["artifacts"].append(
            {
                "kind": "http.file",
                "repository": f"https://models.example/{filename}",
                "revision": f"sha256:{998:064x}",
                "path": str(mount),
            }
        )
        self.module.RUNTIME_SPEC.write_text(json.dumps(document), encoding="utf-8")
        return upscaler

    def test_gemma_assets_are_reassembled_without_the_storage_manifest(self) -> None:
        destination = self.root / "gemma"
        destination.mkdir()
        self.module._link_gemma(destination)
        self.assertEqual(
            {path.name for path in destination.iterdir()},
            set(self.module.GEMMA_FILES.values()),
        )

    def test_named_pipeline_output_contract_is_required(self) -> None:
        pipeline_output = types.SimpleNamespace(
            _fields=self.module.PIPELINE_OUTPUT_FIELDS
        )
        imported = types.SimpleNamespace(PipelineOutput=pipeline_output)
        with (
            mock.patch.object(
                self.module.importlib.metadata,
                "version",
                return_value="1.3.0",
            ),
            mock.patch.object(
                self.module.importlib, "import_module", return_value=imported
            ),
        ):
            self.module._verify_ltx_runtime_contract()

        old_tuple_shape = types.SimpleNamespace(
            PipelineOutput=types.SimpleNamespace(
                _fields=("video", "audio", "num_frames", "tiling_config")
            )
        )
        with (
            mock.patch.object(
                self.module.importlib.metadata,
                "version",
                return_value="1.3.0",
            ),
            mock.patch.object(
                self.module.importlib,
                "import_module",
                return_value=old_tuple_shape,
            ),
            self.assertRaisesRegex(SystemExit, "PipelineOutput contract changed"),
        ):
            self.module._verify_ltx_runtime_contract()

        with (
            mock.patch.object(
                self.module.importlib.metadata,
                "version",
                return_value="1.2.0",
            ),
            mock.patch.object(
                self.module.importlib, "import_module", return_value=imported
            ),
            self.assertRaisesRegex(SystemExit, "runtime version changed"),
        ):
            self.module._verify_ltx_runtime_contract()

    def test_each_checkpoint_selects_the_official_native_pipeline(self) -> None:
        gemma = self.root / "gemma"
        gemma.mkdir()
        self.module._link_gemma(gemma)
        cases = {
            "ltx-2-19b-dev.safetensors": (
                "ltx_pipelines.ti2vid_two_stages",
                "--distilled-lora",
                "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            ),
            "ltx-2-19b-distilled.safetensors": (
                "ltx_pipelines.distilled",
                "--distilled-checkpoint-path",
                "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            ),
            "ltx-2-19b-distilled-fp8.safetensors": (
                "ltx_pipelines.distilled",
                "--quantization",
                "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            ),
            "ltx-2.3-22b-distilled-1.1.safetensors": (
                "ltx_pipelines.distilled",
                "--distilled-checkpoint-path",
                "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            ),
        }
        for filename, (pipeline, required_flag, upscaler_name) in cases.items():
            upscaler = self._upscaler(upscaler_name)
            command = self.module._pipeline_command(
                self._target(filename),
                gemma,
                self.root / "output.mp4",
                7,
                "Operator supplied prompt",
            )
            self.assertIn(pipeline, command)
            self.assertIn(required_flag, command)
            self.assertEqual(
                command[command.index("--spatial-upsampler-path") + 1],
                str(upscaler),
            )
            self.assertFalse(
                any("http://" in value or "https://" in value for value in command)
            )
            self.assertEqual(
                command[command.index("--prompt") + 1], "Operator supplied prompt"
            )
            self.assertEqual(command[command.index("--offload") + 1], "cpu")

    def test_prompt_file_is_required_bounded_utf8_and_read_before_models(self) -> None:
        with self.assertRaisesRegex(SystemExit, "exactly one regular"):
            self.module._load_prompt()
        self.prompt_path.write_text("  Snow, wind, and footsteps.  ", encoding="utf-8")
        self.assertEqual(self.module._load_prompt(), "Snow, wind, and footsteps.")
        (self.module.INPUT_ROOT / "extra.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "exactly one regular"):
            self.module._load_prompt()
        (self.module.INPUT_ROOT / "extra.txt").unlink()
        self.prompt_path.write_bytes(b"\xff")
        with self.assertRaisesRegex(SystemExit, "valid UTF-8"):
            self.module._load_prompt()

        source = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertLess(
            source.index("prompt = _load_prompt()"),
            source.rindex("_target_checkpoint()"),
        )
        self.assertNotIn("VONK_PROMPT", source)

    def test_export_requires_exact_synchronized_media_before_publication(self) -> None:
        output = self.root / "output.mp4"
        output.write_bytes(b"fixture")
        probe = {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 768,
                    "height": 512,
                    "avg_frame_rate": "24/1",
                    "nb_read_frames": "65",
                    "duration": "2.708333",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "24000",
                    "channels": 2,
                    "duration": "2.708333",
                },
            ],
        }
        completed = types.SimpleNamespace(stdout=json.dumps(probe))
        with mock.patch.object(
            self.module.subprocess, "run", return_value=completed
        ) as run:
            self.module._verify_synchronized_mp4(output, 3600, audio_sample_rate=24_000)
        self.assertIn("-count_frames", run.call_args.args[0])

        probe["streams"][0]["codec_name"] = "hevc"
        with (
            mock.patch.object(
                self.module.subprocess,
                "run",
                return_value=types.SimpleNamespace(stdout=json.dumps(probe)),
            ),
            self.assertRaisesRegex(SystemExit, "video properties changed"),
        ):
            self.module._verify_synchronized_mp4(output, 3600, audio_sample_rate=24_000)

        source = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertLess(
            source.index("_verify_synchronized_mp4("), source.rindex("os.replace(")
        )
        self.assertIn(".ltx-synchronized.partial.mp4", source)
        self.assertEqual(
            self.module._expected_audio_sample_rate(
                Path("ltx-2.3-22b-distilled-1.1.safetensors")
            ),
            48_000,
        )
        self.assertEqual(
            self.module._expected_audio_sample_rate(
                Path("ltx-2-19b-distilled.safetensors")
            ),
            24_000,
        )


if __name__ == "__main__":
    unittest.main()
