from __future__ import annotations

import hashlib
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/video/minimax-h3-modular-diffusers"
ADAPTER_PATH = ADAPTER_ROOT / "minimax_h3.py"
RECIPE_PATH = ROOT / "recipes/minimax-h3-diffusers-single.json"
RUNTIME_PATH = ROOT / "runtime-distributions/minimax-h3-modular-diffusers-arm64.json"
MODEL_PATH = ROOT / "model-versions/minimax-h3.json"
DIFFUSERS_REVISION = "efabd60d61c2b7aabf9f182bee6b5b6058980304"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def _canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    content = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _adapter_module():
    module = types.ModuleType("minimax_h3_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(
        compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"),
        module.__dict__,
    )
    return module


class MiniMaxH3AuthorityTests(unittest.TestCase):
    def test_recipe_resolves_exact_local_authorities(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            recipe["model"]["content_sha256"], _canonical_digest(MODEL_PATH)
        )
        self.assertEqual(
            recipe["runtime"]["distribution"]["content_sha256"],
            _canonical_digest(RUNTIME_PATH),
        )
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(runtime["source"]["revision"], DIFFUSERS_REVISION)
        self.assertIsNone(recipe["execution"]["patch_bundle"])
        self.assertEqual(recipe["artifacts"][0]["revision"], MODEL_REVISION)
        self.assertEqual(
            recipe["runtime"]["environment"],
            [
                {"name": "HF_HUB_OFFLINE", "value": "1"},
                {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
            ],
        )

        tags = set(recipe["metadata"]["tags"])
        self.assertIn("candidate", tags)
        self.assertTrue({"video", "audio", "multimodal", "modular-diffusers"} <= tags)
        self.assertTrue(
            {"metadata-only", "non-executable", "integration-required"}.isdisjoint(tags)
        )

    def test_license_fails_closed_for_exact_excluded_territories(self) -> None:
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        restrictions = model["license"]["territorial_restrictions"]
        self.assertEqual(
            restrictions["denied_jurisdictions"],
            ["EU", "GB", "KR", "US"],
        )
        self.assertEqual(
            restrictions["notice"],
            "The MiniMax H3 Community License Agreement excludes the European "
            "Union, United Kingdom, Republic of Korea, and United States of "
            "America from its Applicable Territory.",
        )
        self.assertTrue(model["license"]["operator_acceptance_required"])

    def test_signed_source_bundle_matches_recipe(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        context = recipe["build"]["context"]
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))

    def test_container_and_adapter_forbid_hidden_runtime_downloads(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertIn(f"diffusers.git@{DIFFUSERS_REVISION}", dockerfile)
        self.assertIn(
            f'org.opencontainers.image.revision="{MODEL_REVISION}"', dockerfile
        )
        self.assertGreaterEqual(source.count("local_files_only=True"), 4)
        self.assertIn("pretrained_model_name_or_path=str(MODEL_ROOT)", source)
        self.assertIn(
            'transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"',
            source,
        )
        self.assertIn("**{transformer_name: transformer}", source)
        self.assertNotIn("huggingface.co", source)
        self.assertNotIn("MiniMaxAI/MiniMax-H3", source)

    def test_export_is_atomic_and_joint_audio_video_is_verified(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for required in (
            '"videos"',
            '"audio"',
            '"sampling_rate"',
            "encode_video(",
            "_verify_joint_av(",
            "frame_count=frame_count",
            "audio.sample_rate != 32000",
            'audio.layout.name != "stereo"',
            "os.replace(temporary, destination)",
        ):
            self.assertIn(required, source)

    def test_output_media_contract_is_exact(self) -> None:
        module = _adapter_module()
        video_stream = types.SimpleNamespace(
            width=960,
            height=544,
            average_rate=24,
            codec_context=types.SimpleNamespace(name="h264"),
        )
        audio_stream = types.SimpleNamespace(
            codec_context=types.SimpleNamespace(
                name="aac",
                sample_rate=32_000,
                layout=types.SimpleNamespace(name="stereo"),
            )
        )

        class Container:
            streams = types.SimpleNamespace(video=[video_stream], audio=[audio_stream])

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def decode(self, *, video=None, audio=None):
                if video == 0:
                    return iter([object()] * 124)
                if audio == 0:
                    return iter([types.SimpleNamespace(samples=82_667)] * 2)
                return iter(())

        with mock.patch.dict(
            sys.modules, {"av": types.SimpleNamespace(open=lambda *_a, **_k: Container())}
        ):
            module._verify_joint_av(
                Path("unused.mp4"), width=960, height=544, frame_count=124
            )

        video_stream.codec_context.name = "hevc"
        with mock.patch.dict(
            sys.modules, {"av": types.SimpleNamespace(open=lambda *_a, **_k: Container())}
        ), self.assertRaisesRegex(RuntimeError, "H.264 video and AAC audio"):
            module._verify_joint_av(
                Path("unused.mp4"), width=960, height=544, frame_count=124
            )

    def test_recipe_declares_explicit_smoke_balanced_and_qualification_profiles(
        self,
    ) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        arguments = {
            argument["name"]: argument["value"]
            for argument in recipe["runtime"]["arguments"]
        }
        self.assertEqual(arguments["num-inference-steps"], 31)
        self.assertNotIn("profile", arguments)

        benchmarks = {
            benchmark["name"]: benchmark["configuration"]
            for benchmark in recipe["validation"]["benchmarks"]
        }
        self.assertEqual(
            benchmarks,
            {
                "smoke-only-startup": {
                    "profile": "smoke-only",
                    "sigma_grid_points": 4,
                    "model_evaluations": 3,
                    "smoke_only": True,
                    "timeout_seconds": 14400,
                },
                "balanced-generation": {
                    "profile": "balanced",
                    "sigma_grid_points": 31,
                    "model_evaluations": 30,
                    "timeout_seconds": 28800,
                },
                "qualification-reference": {
                    "profile": "qualification-reference",
                    "sigma_grid_points": 51,
                    "model_evaluations": 50,
                    "timeout_seconds": 43200,
                },
            },
        )

    def test_recipe_declares_truthful_typed_input_slots(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        input_contract = recipe["interfaces"][0]["input"]
        slots = {slot["id"]: slot for slot in input_contract["slots"]}
        self.assertEqual(
            set(slots), {"prompt", "request", "images", "videos", "audio"}
        )
        self.assertEqual(slots["prompt"]["media_types"], ["text/plain"])
        self.assertEqual(slots["prompt"]["min_files"], 1)
        self.assertEqual(slots["request"]["media_types"], ["application/json"])
        self.assertEqual(slots["request"]["max_files"], 1)
        self.assertEqual(slots["request"]["max_file_bytes"], 64 * 1024)
        self.assertEqual(slots["images"]["max_files"], 11)
        self.assertEqual(slots["videos"]["max_files"], 3)
        self.assertEqual(slots["audio"]["max_files"], 3)
        self.assertEqual(slots["request"]["min_files"], 0)
        for slot_id in ("request", "images", "videos", "audio"):
            self.assertEqual(slots[slot_id]["min_files"], 0)
        for slot in slots.values():
            self.assertLessEqual(slot["max_file_bytes"], 512 * 1024 * 1024)
            self.assertLessEqual(slot["max_total_bytes"], 1024 * 1024 * 1024)
        output = recipe["interfaces"][0]["output"]
        self.assertEqual(output["max_total_bytes"], 1024**3)
        self.assertEqual(output["slots"][0]["media_types"], ["video/mp4"])
        self.assertEqual(output["slots"][0]["min_files"], 1)
        self.assertEqual(output["slots"][0]["max_files"], 1)


class MiniMaxH3InputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _adapter_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.module.INPUT_ROOT = Path(self.temporary.name)

    def test_prompt_is_required_and_options_json_is_optional(self) -> None:
        self.assertEqual(self.module._load_request(), {})
        with self.assertRaisesRegex(ValueError, "exactly one UTF-8"):
            self.module._load_prompt()
        (self.module.INPUT_ROOT / "prompt.txt").write_text(
            "  Operator supplied synchronized scene.  ", encoding="utf-8"
        )
        self.assertEqual(
            self.module._load_prompt(), "Operator supplied synchronized scene."
        )

    def test_request_rejects_unknown_fields_and_traversal(self) -> None:
        request = self.module.INPUT_ROOT / "request.json"
        request.write_text('{"unknown": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.module._load_request()
        with self.assertRaisesRegex(ValueError, "inside /inputs"):
            self.module._safe_input_path("../escape.png", "first_image")

    def test_request_accepts_bounded_generation_controls(self) -> None:
        request = self.module.INPUT_ROOT / "request.json"
        request.write_text(
            json.dumps(
                {"num_frames": 124, "width": 960, "height": 544}
            ),
            encoding="utf-8",
        )
        value = self.module._load_request()
        self.assertEqual(self.module._num_frames(value["num_frames"]), 124)
        self.assertEqual(
            self.module._positive_multiple(value["width"], "width", 1344), 960
        )
        self.assertEqual(
            self.module._positive_multiple(value["height"], "height", 768), 544
        )
        with self.assertRaisesRegex(ValueError, "120..345"):
            self.module._num_frames(346)

    def test_profiles_map_named_intent_to_exact_model_evaluations(self) -> None:
        self.assertEqual(
            self.module.PROFILE_SIGMA_GRID_POINTS,
            {
                "smoke-only": 4,
                "balanced": 31,
                "qualification-reference": 51,
            },
        )
        for profile, evaluations in (
            ("smoke-only", 3),
            ("balanced", 30),
            ("qualification-reference", 50),
        ):
            sigma_grid_points = self.module._profile_sigma_grid_points(
                profile, 31
            )
            self.assertEqual(sigma_grid_points - 1, evaluations)

        self.assertEqual(
            self.module._profile_sigma_grid_points(None, 31), 31
        )
        with self.assertRaisesRegex(ValueError, "profile must be one of"):
            self.module._profile_sigma_grid_points("quick", 31)
        with self.assertRaisesRegex(ValueError, "must match a named"):
            self.module._profile_sigma_grid_points(None, 30)

    def test_request_accepts_only_named_profiles(self) -> None:
        request = self.module.INPUT_ROOT / "request.json"
        request.write_text('{"profile": "qualification-reference"}', encoding="utf-8")
        value = self.module._load_request()
        self.assertEqual(
            self.module._profile_sigma_grid_points(value["profile"], 31), 51
        )


if __name__ == "__main__":
    unittest.main()
