from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import ModelDefinition, content_sha256  # noqa: E402
ADAPTER = ROOT / "adapters/video/wan-dancer-diffsynth-disk"
RECIPE = ROOT / "recipes/wan-dancer-14b-disk-offload-pytorch-single.json"
ORIGINAL_RECIPE = ROOT / "recipes/wan-dancer-14b-pytorch-single.json"
RUNTIME = (
    ROOT
    / "runtime-distributions/diffsynth-studio-2-1-5-84f93fc4-disk-cuda13-arm64.json"
)
MODEL = ROOT / "models/wan-dancer-14b.json"
ARCHIVE = (
    ADAPTER
    / "vendor/diffsynth-studio-84f93fc4907b6c193be5501bab0b5c37f383033c.tar.gz"
)
ARCHIVE_SHA256 = "5f0dfef5351341613e2c8ba96806bddb576e5e1f44aaa104c5d5c388cf44bc1b"


def canonical_digest(path: Path) -> str:
    return content_sha256(ModelDefinition.model_validate(json.loads(path.read_text(encoding="utf-8"))))


def load_module(filename: str, name: str):
    path = ADAPTER / filename
    module = types.ModuleType(name)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


class WanDancerDiskOffloadAuthorityTests(unittest.TestCase):
    def test_original_recipe_is_preserved_at_requested_base_digest(self) -> None:
        release = json.loads(
            (ROOT / "recipe-releases/wan-dancer-14b-pytorch-single.json").read_text()
        )
        historical = {entry["version"]: entry["recipe_content_sha256"] for entry in release["history"]}
        self.assertEqual(
            historical["1.0.5"],
            "950296d999671c60362fecb5623d0bf4ad15cbf2d3b6863a9c87ef2cf9fdd940",
        )

    def test_canary_resolves_pinned_model_runtime_and_bounded_memory(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], canonical_digest(MODEL))
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('org.opencontainers.image.revision="84f93fc4907b6c193be5501bab0b5c37f383033c"', dockerfile)
        self.assertIn("diffsynth-studio-84f93fc4907b6c193be5501bab0b5c37f383033c.tar.gz", dockerfile)
        self.assertEqual(json.loads(MODEL.read_text())["source"]["revision"], "85ce88dd8d025459dcf0fe93982d6da8b9002957")
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/modelscope/DiffSynth-Studio/tree/"
            "84f93fc4907b6c193be5501bab0b5c37f383033c",
        )
        self.assertTrue(
            {"experimental", "disk-offload", "physical-oom-gated", "not-default"}
            <= set(recipe["metadata"]["tags"])
        )

        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        envelope = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        ) + memory["system_reserve_bytes"]
        self.assertEqual(envelope, 126_000_000_000)
        self.assertLess(envelope, 126_946_283_520)

        benchmark = recipe["validation"]["benchmarks"][0]["configuration"]
        self.assertEqual(benchmark["maximum_generation_pixels"], 921_600)
        self.assertEqual(benchmark["device_residency_threshold_gib"], 64)
        self.assertIs(benchmark["stage_process_isolation"], True)

    def test_vendored_diffsynth_source_is_exact_and_offline_patch_applies(self) -> None:
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), ARCHIVE_SHA256)
        runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
        self.assertEqual(
            runtime["source"]["revision"],
            "84f93fc4907b6c193be5501bab0b5c37f383033c",
        )
        self.assertEqual(runtime["source"]["archive_sha256"], ARCHIVE_SHA256)

        with tempfile.TemporaryDirectory() as value:
            source_root = Path(value) / "source"
            source_root.mkdir()
            with tarfile.open(ARCHIVE, "r:gz") as source:
                source.extractall(source_root, filter="data")
            unpacked = next(source_root.iterdir())
            subprocess.run(
                ["python3", str(ADAPTER / "patch-runtime.py"), str(unpacked)],
                check=True,
            )
            config = (unpacked / "diffsynth/core/loader/config.py").read_text()
            data_init = (unpacked / "diffsynth/core/data/__init__.py").read_text()
            self.assertNotIn("from modelscope import", config)
            self.assertNotIn("from huggingface_hub import", config)
            self.assertIn("network model loading is disabled", config)
            self.assertNotIn("UnifiedDataset", data_init)

    def test_runtime_uses_disk_mapping_process_isolation_and_no_downloads(self) -> None:
        generator = (ADAPTER / "generate.py").read_text(encoding="utf-8")
        runner = (ADAPTER / "run.py").read_text(encoding="utf-8")
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('"offload_dtype": "disk"', generator)
        self.assertIn('"offload_device": "disk"', generator)
        self.assertIn("VRAM_LIMIT_GIB = 64.0", generator)
        self.assertIn('"--stage",\n            "global"', runner)
        self.assertIn('"--stage",\n            "local"', runner)
        self.assertIn("DIFFSYNTH_SKIP_DOWNLOAD=True", dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertNotIn("git clone", dockerfile)


class WanDancerDiskOffloadInputTests(unittest.TestCase):
    def test_generation_canvas_accepts_720p_and_rejects_a_larger_canvas(self) -> None:
        runner = load_module("run.py", "wan_dancer_disk_runner")
        self.assertEqual(
            runner._generation_dimensions({"height": 720, "width": 1280}),
            (720, 1280),
        )
        self.assertEqual(
            runner._generation_dimensions({"height": 1280, "width": 720}),
            (1280, 720),
        )
        with self.assertRaisesRegex(ValueError, "at most 921600 pixels"):
            runner._generation_dimensions({"height": 1280, "width": 1280})

    def test_local_keyframes_preserve_native_leading_frame_mapping(self) -> None:
        generator = load_module("generate.py", "wan_dancer_disk_generator")

        class Image:
            @staticmethod
            def new(_mode, _size, _color):
                return "black"

        class VideoData:
            def __init__(self, **_kwargs):
                self.frames = list(range(149))

            def __len__(self):
                return len(self.frames)

            def __getitem__(self, index):
                return self.frames[index]

        modules = {
            "PIL": SimpleNamespace(Image=Image),
            "diffsynth": types.ModuleType("diffsynth"),
            "diffsynth.utils": types.ModuleType("diffsynth.utils"),
            "diffsynth.utils.data": SimpleNamespace(VideoData=VideoData),
        }
        with mock.patch.dict(sys.modules, modules):
            frames, mask = generator._local_keyframes(
                Path("global.mp4"), 24, 1280, 720
            )
        self.assertEqual(frames[:24], list(range(24)))
        self.assertEqual(frames[24:], ["black"] * 125)
        self.assertEqual(mask, [1] * 24 + [0] * 125)


if __name__ == "__main__":
    unittest.main()
