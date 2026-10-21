from __future__ import annotations

import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ltx_protocol import assert_streaming_quantization_supported, manifest_bytes

ROOT = Path(__file__).resolve().parents[1]
RECIPE_SLUGS = (
    "ltx-2-19b-dev-bf16-diffusers-single",
    "ltx-2-19b-distilled-diffusers-single",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class LtxSyncAuthorityTests(unittest.TestCase):
    def test_all_sync_checkpoint_commands_use_supported_streaming_policies(
        self,
    ) -> None:
        # LTX 1.5.0 diffusion_stage.py:150-202 selects streaming for CPU AND disk
        # offload, then rejects every quantization fuse rule except fp8_cast.
        for adapter in ("ltx2-sync-native", "ltx23-sync-native-disk"):
            module = runpy.run_path(str(ROOT / f"adapters/video/{adapter}/run.py"))
            for checkpoint in module["TARGET_FILENAMES"]:
                with self.subTest(adapter=adapter, checkpoint=checkpoint):
                    command_builder = module["_pipeline_command"]
                    with mock.patch.dict(
                        command_builder.__globals__,
                        {
                            "_spatial_upscaler": lambda *_: Path("/models/upscaler"),
                            "_single_artifact": lambda *_: Path("/models/lora"),
                        },
                    ):
                        command = command_builder(
                            Path("/models/target") / checkpoint,
                            Path("/models/gemma"),
                            Path("/outputs/video.mp4"),
                            7,
                            "Prompt",
                            [],
                        )
                    assert_streaming_quantization_supported(command)
                    if checkpoint == "ltx-2-19b-distilled-fp8.safetensors":
                        self.assertNotIn("--offload", command)
                        self.assertEqual(
                            command[command.index("--quantization") + 1],
                            "fp8-scaled-mm",
                        )

    def test_streaming_guard_rejects_incompatible_quantization_policies(self) -> None:
        for offload in ("cpu", "disk"):
            for policy in ("fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"):
                with (
                    self.subTest(offload=offload, policy=policy),
                    self.assertRaisesRegex(AssertionError, "block streaming"),
                ):
                    assert_streaming_quantization_supported(
                        ["--offload", offload, "--quantization", policy]
                    )
            for policy_args in ([], ["--quantization", "fp8-cast"]):
                assert_streaming_quantization_supported(
                    ["--offload", offload, *policy_args]
                )
        assert_streaming_quantization_supported(["--quantization", "fp8-scaled-mm"])
        assert_streaming_quantization_supported(
            ["--offload", "none", "--quantization", "fp8-scaled-mm"]
        )

    def test_recipes_resolve_exact_runnable_authorities_and_closure(self) -> None:
        for slug in RECIPE_SLUGS:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertTrue(recipe["models"])
            self.assertEqual(recipe["runtime"]["engine"], "pytorch-pipeline")
            self.assertEqual(recipe["interfaces"][0]["adapter"], "video-job")
            self.assertEqual(recipe["topology"]["node_count"], 1)

    def test_container_is_pinned_and_runtime_is_offline(self) -> None:
        for path in (
            ROOT / "adapters/video/ltx23-sync-native-disk/Dockerfile",
            ROOT / "adapters/video/ltx2-sync-native/Dockerfile",
        ):
            self.assertIn("@sha256:", path.read_text())
        for slug in RECIPE_SLUGS:
            environment = {
                item["name"]
                for item in load(ROOT / "recipes" / f"{slug}.json")["runtime"][
                    "environment"
                ]
            }
            self.assertIn("HF_HUB_OFFLINE", environment)

    def test_signed_source_bundle_matches_each_recipe_context(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for slug in RECIPE_SLUGS:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            context = recipe["execution"]["build"]["context"]
            _, _, digest = tool["source_bundle"](ROOT / context["path"])
            self.assertTrue(digest)

    def test_runtime_output_contract_rejects_wrong_streams_and_keeps_atomic_publish(
        self,
    ) -> None:
        module = runpy.run_path(
            str(ROOT / "adapters/video/ltx23-sync-native-disk/run.py")
        )
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
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.mp4"
            output.write_bytes(b"fixture")
            with mock.patch.object(
                module["subprocess"],
                "run",
                return_value=types.SimpleNamespace(stdout=json.dumps(probe)),
            ):
                module["_verify_synchronized_mp4"](
                    output, 3600, audio_sample_rate=24_000
                )
            probe["streams"][0]["codec_name"] = "hevc"
            with (
                mock.patch.object(
                    module["subprocess"],
                    "run",
                    return_value=types.SimpleNamespace(stdout=json.dumps(probe)),
                ),
                self.assertRaisesRegex(SystemExit, "video properties changed"),
            ):
                module["_verify_synchronized_mp4"](
                    output, 3600, audio_sample_rate=24_000
                )
        source = (ROOT / "adapters/video/ltx23-sync-native-disk/run.py").read_text()
        self.assertLess(
            source.index("_verify_synchronized_mp4("), source.index("os.replace(")
        )
        self.assertIn(".ltx-synchronized.partial.mp4", source)

    def test_prompt_is_bounded_before_runtime_and_gemma_reassembly_is_explicit(
        self,
    ) -> None:
        module = runpy.run_path(
            str(ROOT / "adapters/video/ltx23-sync-native-disk/run.py")
        )
        with tempfile.TemporaryDirectory() as directory:
            module["_load_prompt"].__globals__["INPUT_ROOT"] = Path(directory)

            def stage_prompt(text: str) -> None:
                payload = text.encode("utf-8")
                (Path(directory) / "prompt.txt").write_bytes(payload)
                (Path(directory) / "manifest.json").write_bytes(
                    manifest_bytes(payload, name="prompt.txt")
                )

            stage_prompt("  bounded prompt  ")
            self.assertEqual(module["_load_prompt"](), "bounded prompt")
            stage_prompt("x" * 4097)
            with self.assertRaisesRegex(SystemExit, "1..4096"):
                module["_load_prompt"]()
        self.assertEqual(len(module["GEMMA_FILES"]), 22)

    def test_all_native_runners_match_the_ltx_output_contract(self) -> None:
        expected = ("video", "audio", "num_frames", "tiling_config")
        adapters = (
            "adapters/video/ltx2-pytorch/pipelines/run.py",
            "adapters/video/ltx2-sync-native/run.py",
            "adapters/video/ltx23-sync-native-disk/run.py",
        )
        for relative in adapters:
            module = runpy.run_path(str(ROOT / relative))
            self.assertRegex(module["LTX_PIPELINES_VERSION"], r"^\d+\.\d+\.\d+$")
            self.assertEqual(module["PIPELINE_OUTPUT_FIELDS"], expected)
            pipeline = types.SimpleNamespace(_fields=expected)
            with (
                mock.patch.object(
                    module["importlib"].metadata,
                    "version",
                    return_value=module["LTX_PIPELINES_VERSION"],
                ),
                mock.patch.object(
                    module["importlib"],
                    "import_module",
                    return_value=types.SimpleNamespace(PipelineOutput=pipeline),
                ),
            ):
                module["_verify_ltx_runtime_contract"]()
            pipeline._fields = (*expected, "video_latent")
            with (
                mock.patch.object(
                    module["importlib"].metadata,
                    "version",
                    return_value=module["LTX_PIPELINES_VERSION"],
                ),
                mock.patch.object(
                    module["importlib"],
                    "import_module",
                    return_value=types.SimpleNamespace(PipelineOutput=pipeline),
                ),
                self.assertRaisesRegex(SystemExit, "PipelineOutput contract changed"),
            ):
                module["_verify_ltx_runtime_contract"]()


if __name__ == "__main__":
    unittest.main()


def test_smoke_frames_reach_native_command_and_verifier(tmp_path: Path):
    module = runpy.run_path(str(ROOT / "adapters/video/ltx2-sync-native/run.py"))
    builder = module["_pipeline_command"]
    with mock.patch.dict(
        builder.__globals__, {"_spatial_upscaler": lambda *_: Path("/models/upscaler")}
    ):
        command = builder(
            Path("/models/ltx-2-19b-distilled.safetensors"),
            tmp_path,
            tmp_path / "video.mp4",
            0,
            "Prompt",
            [],
            frames=1,
        )
    assert command[command.index("--num-frames") + 1] == "1"
    probe = {
        "format": {"format_name": "mp4"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 768,
                "height": 512,
                "avg_frame_rate": "24/1",
                "nb_read_frames": "1",
                "duration": str(1 / 24),
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "24000",
                "channels": 2,
                "duration": str(1 / 24),
            },
        ],
    }
    output = tmp_path / "video.mp4"
    output.write_bytes(b"fixture")
    with mock.patch.object(
        module["subprocess"],
        "run",
        return_value=types.SimpleNamespace(stdout=json.dumps(probe)),
    ):
        module["_verify_synchronized_mp4"](
            output, 600, audio_sample_rate=24000, frames=1
        )
