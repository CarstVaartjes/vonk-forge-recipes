from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/video/ltx2-sync-native"
ADAPTER_PATH = ADAPTER_ROOT / "run.py"
RUNTIME_PATH = ROOT / "runtime-distributions/ltx2-pipelines-1-2-arm64.json"
GEMMA_PATH = ROOT / "model-versions/ltx-2-gemma3-text-encoder-dfcc2108.json"
SOURCE_REVISION = "400fd31054597515f47125691032c04b1c3ee24e"
MODEL_SPECS = {
    "ltx-2-19b-dev-bf16-diffusers-single": "ltx-2-19b-dev-bf16.json",
    "ltx-2-19b-distilled-diffusers-single": "ltx-2-19b-distilled.json",
    "ltx-2-19b-distilled-fp8-diffusers-single": "ltx-2-19b-distilled-fp8.json",
    "ltx-2-3-22b-distilled-1-1-diffusers-single": "ltx-2-3-22b-distilled-1-1.json",
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
        self.assertIn(f"LTX-2/archive/{SOURCE_REVISION}.tar.gz", dockerfile)
        self.assertIn(runtime["source"]["archive_sha256"], dockerfile)
        self.assertIn("sha256sum --check --strict", dockerfile)
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

    def test_gemma_assets_are_reassembled_without_the_storage_manifest(self) -> None:
        destination = self.root / "gemma"
        destination.mkdir()
        self.module._link_gemma(destination)
        self.assertEqual(
            {path.name for path in destination.iterdir()},
            set(self.module.GEMMA_FILES.values()),
        )

    def test_each_checkpoint_selects_the_official_native_pipeline(self) -> None:
        gemma = self.root / "gemma"
        gemma.mkdir()
        self.module._link_gemma(gemma)
        cases = {
            "ltx-2-19b-dev.safetensors": (
                "ltx_pipelines.ti2vid_two_stages",
                "--distilled-lora",
            ),
            "ltx-2-19b-distilled.safetensors": (
                "ltx_pipelines.distilled",
                "--distilled-checkpoint-path",
            ),
            "ltx-2-19b-distilled-fp8.safetensors": (
                "ltx_pipelines.distilled",
                "--quantization",
            ),
            "ltx-2.3-22b-distilled-1.1.safetensors": (
                "ltx_pipelines.distilled",
                "--distilled-checkpoint-path",
            ),
        }
        for filename, (pipeline, required_flag) in cases.items():
            command = self.module._pipeline_command(
                self._target(filename), gemma, self.root / "output.mp4", 7
            )
            self.assertIn(pipeline, command)
            self.assertIn(required_flag, command)
            self.assertFalse(
                any("http://" in value or "https://" in value for value in command)
            )

    def test_export_requires_both_video_and_audio_streams(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertIn('if not {"video", "audio"}.issubset(streams):', source)
        self.assertIn("ffprobe", source)
        self.assertIn("ltx-synchronized.mp4", source)


if __name__ == "__main__":
    unittest.main()
