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
ADAPTER = ROOT / "adapters/video/wan-dancer-native"
RECIPE = ROOT / "recipes/wan-dancer-14b-pytorch-single.json"
RUNTIME = ROOT / "runtime-distributions/wan-dancer-native-e6c87a9-cuda13-arm64.json"
MODEL = ROOT / "model-versions/wan-dancer-14b.json"
ARCHIVE_SHA256 = "92c529d7727c75c6515ea990d27883a45bf566587cc9f5d325a0a488b9fa1649"


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_runner():
    path = ADAPTER / "run.py"
    module = types.ModuleType("wan_dancer_runner")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


class WanDancerAuthorityTests(unittest.TestCase):
    def test_recipe_resolves_complete_immutable_authorities(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(MODEL))
        self.assertEqual(
            recipe["runtime"]["distribution"]["content_sha256"],
            canonical_digest(RUNTIME),
        )
        self.assertEqual(
            recipe["artifacts"][0]["revision"],
            "85ce88dd8d025459dcf0fe93982d6da8b9002957",
        )
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/Wan-Video/Wan-Dancer/tree/"
            "e6c87a94ec733230dac15b924c015f6e6501e618",
        )
        self.assertEqual(recipe["topology"]["node_count"], 1)
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(memory["startup_peak_bytes"], 120000000000)
        self.assertEqual(memory["steady_state_bytes"], 110000000000)
        self.assertEqual(memory["runtime_growth_bytes"], 8000000000)
        self.assertEqual(memory["system_reserve_bytes"], 8000000000)
        self.assertIn(
            {"source": "inputs", "target": "/inputs", "read_only": True},
            recipe["runtime"]["security"]["mounts"],
        )
        interface = recipe["interfaces"][0]
        slots = {slot["id"]: slot for slot in interface["input"]["slots"]}
        self.assertEqual(set(slots), {"prompt", "reference-image", "music", "controls"})
        self.assertEqual(
            (slots["prompt"]["min_files"], slots["prompt"]["max_files"]), (1, 1)
        )
        self.assertEqual(
            (slots["controls"]["min_files"], slots["controls"]["max_files"]), (0, 1)
        )
        output = interface["output"]
        self.assertEqual(output["max_total_bytes"], 536870912)
        self.assertEqual(output["slots"][0]["media_types"], ["video/mp4"])
        self.assertIn("H.264/AAC", output["slots"][0]["description"])
        configuration = recipe["validation"]["benchmarks"][0]["configuration"]
        self.assertEqual(configuration["video_codec"], "h264")
        self.assertEqual(configuration["audio_codec"], "aac")
        self.assertEqual(configuration["audio_sample_rate"], 44100)
        self.assertEqual(configuration["maximum_generation_pixels"], 921600)
        self.assertEqual(
            configuration["output_frame_rule"],
            "floor((music-duration-seconds-0.2)*30)",
        )

    def test_vendored_source_is_bounded_and_fail_closed_patch_applies(self) -> None:
        parts = sorted((ADAPTER / "vendor").glob("*.part-*"))
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(part.stat().st_size < 12 * 1024 * 1024 for part in parts))
        archive = b"".join(part.read_bytes() for part in parts)
        self.assertEqual(hashlib.sha256(archive).hexdigest(), ARCHIVE_SHA256)

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            archive_path = root / "source.tar.gz"
            archive_path.write_bytes(archive)
            with tarfile.open(archive_path, "r:gz") as source:
                source.extractall(root / "unpacked", filter="data")
            source_root = next((root / "unpacked").iterdir())
            subprocess.run(
                ["python3", str(ADAPTER / "patch-upstream.py"), str(source_root)],
                check=True,
            )
            global_stage = (source_root / "gen_video/gen_video_global.py").read_text()
            local_stage = (source_root / "gen_video/gen_video_local.py").read_text()
            self.assertIn(
                'ModelConfig(path="/models/global_model.safetensors"', global_stage
            )
            self.assertIn(
                'ModelConfig(path="/models/local_model.safetensors"', local_stage
            )
            self.assertNotIn("use_usp=True", global_stage + local_stage)
            self.assertNotIn("dist.get_rank()", global_stage + local_stage)
            self.assertIn("cfg_scale=args.cfg_scale", local_stage)

    def test_runtime_has_no_model_download_path(self) -> None:
        runner = (ADAPTER / "run.py").read_text(encoding="utf-8")
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("snapshot_download", runner)
        self.assertNotIn("from_pretrained", runner)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)


class WanDancerInputTests(unittest.TestCase):
    def test_generation_canvas_accepts_portrait_and_landscape_720p(self) -> None:
        runner = load_runner()

        self.assertEqual(
            runner._generation_dimensions({"height": 720, "width": 1280}),
            (720, 1280),
        )
        self.assertEqual(
            runner._generation_dimensions({"height": 1280, "width": 720}),
            (1280, 720),
        )

    def test_generation_canvas_rejects_1280_square(self) -> None:
        runner = load_runner()

        with self.assertRaisesRegex(ValueError, "at most 921600 pixels"):
            runner._generation_dimensions({"height": 1280, "width": 1280})

    def test_expected_output_is_derived_from_reference_aspect_and_music(self) -> None:
        runner = load_runner()

        class Source:
            size = (64, 64)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def verify(self):
                return None

        pil = SimpleNamespace(
            Image=SimpleNamespace(open=lambda _path: Source()),
            UnidentifiedImageError=ValueError,
        )
        with mock.patch.dict(sys.modules, {"PIL": pil}):
            output = runner._expected_output(
                Path("person.png"),
                target_height=1280,
                target_width=720,
                music_duration=1.0,
            )
        self.assertEqual(output, (720, 720, 24, 0.8))

    def test_output_requires_derived_h264_aac_media_contract(self) -> None:
        runner = load_runner()
        probe = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "0.800000",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 720,
                    "height": 720,
                    "avg_frame_rate": "30/1",
                    "nb_read_frames": "24",
                    "duration": "0.800000",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "44100",
                    "duration": "0.800000",
                },
            ],
        }
        completed = SimpleNamespace(stdout=json.dumps(probe))
        with mock.patch.object(runner.subprocess, "run", return_value=completed) as run:
            runner._verify_output(
                Path("dance.mp4"),
                60,
                expected_width=720,
                expected_height=720,
                expected_frames=24,
                expected_duration=0.8,
            )
        self.assertIn("-count_frames", run.call_args.args[0])

        invalid_cases = (
            (("format", "format_name"), "matroska", "MP4 container"),
            (("streams", 0, "codec_name"), "vp9", "H.264 video"),
            (("streams", 1, "codec_name"), "opus", "codec must be AAC"),
            (("streams", 1, "sample_rate"), "48000", "must be 44100 Hz"),
            (("streams", 0, "width"), 704, "dimensions"),
            (("streams", 0, "avg_frame_rate"), "29/1", "must be 30 fps"),
            (("streams", 0, "nb_read_frames"), "23", "exactly 24 frames"),
            (("streams", 0, "duration"), "0.6", "video duration"),
            (("streams", 1, "duration"), "1.1", "audio duration"),
            (("format", "duration"), "1.1", "MP4 duration"),
        )
        for path, value, message in invalid_cases:
            with self.subTest(path=path):
                invalid = json.loads(json.dumps(probe))
                target = invalid
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
                with (
                    mock.patch.object(
                        runner.subprocess,
                        "run",
                        return_value=SimpleNamespace(stdout=json.dumps(invalid)),
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    runner._verify_output(
                        Path("dance.mp4"),
                        60,
                        expected_width=720,
                        expected_height=720,
                        expected_frames=24,
                        expected_duration=0.8,
                    )

    def test_request_requires_image_music_prompt_and_rejects_traversal(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as value:
            inputs = Path(value)
            runner.INPUT_ROOT = inputs
            (inputs / "person.jpg").write_bytes(b"jpeg")
            (inputs / "music.wav").write_bytes(b"wav")
            (inputs / "request.json").write_text(
                json.dumps(
                    {
                        "reference_image": "person.jpg",
                        "music": "music.wav",
                        "prompt": "A precise K-pop dance",
                        "style": "k-pop",
                    }
                ),
                encoding="utf-8",
            )
            request = runner._request()
            self.assertEqual(request["style"], "k-pop")
            self.assertEqual(
                runner._input_file(
                    request["reference_image"], "reference_image", runner.IMAGE_SUFFIXES
                ),
                inputs / "person.jpg",
            )
            with self.assertRaises(ValueError):
                runner._input_file(
                    "../person.jpg", "reference_image", runner.IMAGE_SUFFIXES
                )

    def test_manifest_slots_supply_prompt_image_music_and_optional_controls(
        self,
    ) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as value:
            inputs = Path(value)
            runner.INPUT_ROOT = inputs
            files = {
                "prompt": inputs / "prompt.txt",
                "reference-image": inputs / "person.png",
                "music": inputs / "music.wav",
                "controls": inputs / "controls.json",
            }
            files["prompt"].write_text("A precise street dance", encoding="utf-8")
            files["reference-image"].write_bytes(b"png")
            files["music"].write_bytes(b"wav")
            files["controls"].write_text(
                json.dumps({"style": "street", "num_inference_steps_global": 2}),
                encoding="utf-8",
            )
            entries = []
            total = 0
            media_types = {
                "prompt": "text/plain",
                "reference-image": "image/png",
                "music": "audio/wav",
                "controls": "application/json",
            }
            for slot, path in files.items():
                payload = path.read_bytes()
                total += len(payload)
                entries.append(
                    {
                        "slot": slot,
                        "name": path.name,
                        "media_type": media_types[slot],
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            (inputs / "manifest.json").write_text(
                json.dumps(
                    {"schema_version": 1, "total_bytes": total, "files": entries}
                ),
                encoding="utf-8",
            )
            manifest = runner._manifest()
            request = runner._request(manifest)
            self.assertEqual(request["prompt"], "A precise street dance")
            self.assertEqual(request["reference_image"], "person.png")
            self.assertEqual(request["music"], "music.wav")
            self.assertEqual(request["style"], "street")
            self.assertEqual(request["num_inference_steps_global"], 2)

    def test_manifest_rejects_unknown_slots_and_mismatched_media_types(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as value:
            inputs = Path(value)
            runner.INPUT_ROOT = inputs
            prompt = inputs / "prompt.txt"
            prompt.write_text("Dance", encoding="utf-8")

            def write_manifest(slot: str, media_type: str) -> None:
                payload = prompt.read_bytes()
                (inputs / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "total_bytes": len(payload),
                            "files": [
                                {
                                    "slot": slot,
                                    "name": prompt.name,
                                    "media_type": media_type,
                                    "size_bytes": len(payload),
                                    "sha256": hashlib.sha256(payload).hexdigest(),
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            write_manifest("extra", "text/plain")
            with self.assertRaisesRegex(ValueError, "unsupported slot"):
                runner._manifest()
            write_manifest("prompt", "application/json")
            with self.assertRaisesRegex(ValueError, "prompt slot contract"):
                runner._manifest()


if __name__ == "__main__":
    unittest.main()
