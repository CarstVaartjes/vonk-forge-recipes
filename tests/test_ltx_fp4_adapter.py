# ruff: noqa: S102 -- isolated synthetic adapter module is executed in tests.
from __future__ import annotations

import hashlib
import json
import runpy
import shlex
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from documents import display_of
from generated_catalog import GENERATED
from ltx_protocol import assert_streaming_quantization_supported, manifest_bytes

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/video/ltx2-pytorch"
ADAPTER_PATH = ADAPTER_ROOT / "pipelines/run.py"
RECIPE = ROOT / "recipes/ltx-2-19b-dev-fp4-pytorch-single.json"


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            _document(path), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _adapter_module():
    module = types.ModuleType("ltx2_fp4_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(
        compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"),
        module.__dict__,
    )
    return module


class LtxFp4PromptContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _adapter_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.adapter.INPUT_ROOT = Path(self.temporary.name) / "inputs"
        self.adapter.INPUT_ROOT.mkdir()
        self.adapter.MODEL_ROOT = Path(self.temporary.name) / "models"
        self.adapter.MODEL_ROOT.mkdir()
        for relative in self.adapter.GEMMA_REQUIRED_PATHS:
            path = self.adapter.MODEL_ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.prompt_path = self.adapter.INPUT_ROOT / "scene.txt"

    def stage_prompt(self, payload: bytes) -> None:
        self.prompt_path.write_bytes(payload)
        (self.adapter.INPUT_ROOT / "manifest.json").write_bytes(
            manifest_bytes(payload, name=self.prompt_path.name)
        )

    def test_gemma_loader_root_contains_encoder_and_tokenizer_closure(self) -> None:
        root = self.adapter._gemma_root()
        self.assertEqual(root, self.adapter.MODEL_ROOT)
        self.assertTrue((root / "text_encoder/model.safetensors.index.json").is_file())
        self.assertTrue((root / "tokenizer/tokenizer.model").is_file())
        self.assertNotEqual(root, root / "text_encoder")

        (root / "tokenizer/tokenizer.model").unlink()
        with self.assertRaisesRegex(SystemExit, "tokenizer/tokenizer.model"):
            self.adapter._gemma_root()

    def test_prompt_must_be_exactly_one_bounded_utf8_file(self) -> None:
        with self.assertRaisesRegex(SystemExit, "manifest"):
            self.adapter._load_prompt()
        self.stage_prompt(b"  A precise operator prompt.  ")
        self.assertEqual(self.adapter._load_prompt(), "A precise operator prompt.")
        self.stage_prompt(b"\xff")
        with self.assertRaisesRegex(SystemExit, "valid UTF-8"):
            self.adapter._load_prompt()

    def test_main_passes_prompt_without_environment_fallback(self) -> None:
        self.stage_prompt(b"Synchronized fox scene")
        output = Path(self.temporary.name) / "outputs"
        argv = [
            "run.py",
            "--entrypoint",
            "/opt/vonk/source/pipelines/run.py",
            "--output-mime",
            "video/mp4",
            "--output-dir",
            str(output),
            "--num-frames",
            "1",
            "--num-inference-steps",
            "2",
        ]

        def run_pipeline(command, **_kwargs):
            # Reproduce the captured failure before the adapter can publish output:
            # a plain streaming model has no NVFP4 companion parameters.
            if command[command.index("--offload") + 1] != "none":
                raise KeyError("attn1.to_k.input_scale")
            self.assertEqual(
                command[command.index("--quantization") + 1], "nvfp4-prequant"
            )
            output_path = Path(command[command.index("--output-path") + 1])
            output_path.write_bytes(b"fixture")

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(self.adapter, "_verify_ltx_runtime_contract"),
            mock.patch.object(
                self.adapter.subprocess, "run", side_effect=run_pipeline
            ) as run,
            mock.patch.object(self.adapter, "_verify_synchronized_mp4") as verify,
        ):
            self.adapter.main()
        command = run.call_args_list[0].args[0]
        assert_streaming_quantization_supported(command)
        self.assertEqual(
            command[command.index("--prompt") + 1], "Synchronized fox scene"
        )
        self.assertEqual(
            Path(command[command.index("--gemma-root") + 1]),
            self.adapter.MODEL_ROOT,
        )
        self.assertTrue(
            (
                Path(command[command.index("--gemma-root") + 1])
                / "tokenizer/tokenizer.model"
            ).is_file()
        )
        for flag, expected in (
            ("--width", "768"),
            ("--height", "512"),
            ("--num-frames", "1"),
            ("--num-inference-steps", "2"),
            ("--frame-rate", "24"),
            ("--offload", "none"),
            ("--quantization", "nvfp4-prequant"),
            ("--max-batch-size", "1"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertEqual(
            Path(command[command.index("--output-path") + 1]).name,
            ".ltx2.partial.mp4",
        )
        verify.assert_called_once()
        self.assertTrue((output / "ltx2.mp4").is_file())
        self.assertNotIn("VONK_PROMPT", ADAPTER_PATH.read_text(encoding="utf-8"))

    def test_prequantized_checkpoint_requires_native_loader(self) -> None:
        checkpoint = ["--checkpoint-path", "/models/ltx-2-19b-dev-fp4.safetensors"]
        for mode in ("cpu", "disk", "none"):
            for policy in ([], ["--quantization", "fp8-cast"]):
                with (
                    self.subTest(mode=mode, policy=policy),
                    self.assertRaisesRegex(AssertionError, "NVFP4 checkpoints"),
                ):
                    assert_streaming_quantization_supported(
                        [*checkpoint, "--offload", mode, *policy]
                    )
        for mode in ("cpu", "disk"):
            with self.assertRaisesRegex(AssertionError, "NVFP4 checkpoints"):
                assert_streaming_quantization_supported(
                    [*checkpoint, "--offload", mode, "--quantization", "nvfp4-prequant"]
                )
        assert_streaming_quantization_supported(
            [*checkpoint, "--offload", "none", "--quantization", "nvfp4-prequant"]
        )

    def test_native_nvfp4_kernels_are_built_and_checked_for_spark(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text()
        self.assertIn("13.0.2-devel-ubuntu24.04@sha256:", dockerfile)
        self.assertIn('TORCH_CUDA_ARCH_LIST="12.0 12.1" MAX_JOBS=2', dockerfile)
        self.assertIn(
            "--no-deps /opt/vonk/source/ltx2/packages/ltx-kernels", dockerfile
        )
        self.assertIn("import torch; import nvfp4_cpp", dockerfile)
        self.assertIn("cuobjdump --list-elf", dockerfile)
        self.assertIn("grep --quiet sm_121a", dockerfile)
        self.assertIn("--fuzz=0", dockerfile)
        patch = (ADAPTER_ROOT / "nvfp4-sm121.patch").read_text()
        self.assertIn('+_NVFP4_ARCHES = ["100a", "110a", "120a", "121a"]', patch)
        self.assertIn('+        elif tok.startswith("12.1"):', patch)
        self.assertIn('+            sel.append("121a")', patch)

    def test_cuda_compiler_mismatch_fails_before_kernel_build(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text()
        command = next(
            line.removeprefix("RUN ")
            for line in dockerfile.splitlines()
            if line.startswith("RUN python3 -c") and "torch.version.cuda" in line
        )
        self.assertLess(
            dockerfile.index(command), dockerfile.index("packages/ltx-kernels")
        )
        program = shlex.split(command)[2]
        torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda="13.0"))
        for version in ("13.0", "13.4"):
            with (
                self.subTest(version=version),
                mock.patch.dict(sys.modules, {"torch": torch}),
                mock.patch(
                    "subprocess.check_output",
                    return_value=f"Cuda compilation tools, release {version}, V{version}.88",
                ),
            ):
                if version == "13.0":
                    exec(program, {})
                else:
                    with self.assertRaisesRegex(AssertionError, "differs from torch"):
                        exec(program, {})

    def test_named_pipeline_output_contract_is_required(self) -> None:
        pipeline_output = types.SimpleNamespace(
            _fields=self.adapter.PIPELINE_OUTPUT_FIELDS
        )
        imported = types.SimpleNamespace(PipelineOutput=pipeline_output)
        with (
            mock.patch.object(
                self.adapter.importlib.metadata,
                "version",
                return_value=self.adapter.LTX_PIPELINES_VERSION,
            ),
            mock.patch.object(
                self.adapter.importlib, "import_module", return_value=imported
            ),
        ):
            self.adapter._verify_ltx_runtime_contract()

        imported.PipelineOutput._fields = (
            "video",
            "audio",
            "num_frames",
            "tiling_config",
            "keyframes",
        )
        with (
            mock.patch.object(
                self.adapter.importlib.metadata,
                "version",
                return_value=self.adapter.LTX_PIPELINES_VERSION,
            ),
            mock.patch.object(
                self.adapter.importlib, "import_module", return_value=imported
            ),
            self.assertRaisesRegex(SystemExit, "PipelineOutput contract changed"),
        ):
            self.adapter._verify_ltx_runtime_contract()

    def test_output_media_contract_is_exact(self) -> None:
        output = Path(self.temporary.name) / "output.mp4"
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
                    "nb_read_frames": "97",
                    "duration": "4.041667",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "24000",
                    "channels": 2,
                    "duration": "4.041667",
                },
            ],
        }
        with mock.patch.object(
            self.adapter.subprocess,
            "run",
            return_value=types.SimpleNamespace(stdout=json.dumps(probe)),
        ):
            self.adapter._verify_synchronized_mp4(output, 3600)
        probe["streams"][1]["channels"] = 1
        with (
            mock.patch.object(
                self.adapter.subprocess,
                "run",
                return_value=types.SimpleNamespace(stdout=json.dumps(probe)),
            ),
            self.assertRaisesRegex(SystemExit, "audio properties changed"),
        ):
            self.adapter._verify_synchronized_mp4(output, 3600)

        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("        ffmpeg \\\n", dockerfile)

    def test_recipe_package_binds_prompt_slot_and_source_bundle(self) -> None:
        recipe = _document(RECIPE)
        prompt_input = recipe["interfaces"][0]["input"]
        self.assertTrue(prompt_input["required"])
        self.assertEqual(prompt_input["media_types"], ["text/plain"])
        self.assertEqual(prompt_input["max_bytes"], 16 * 1024)
        self.assertEqual(prompt_input["slots"][0]["min_files"], 1)
        self.assertEqual(prompt_input["slots"][0]["max_files"], 1)
        output = recipe["interfaces"][0]["output"]
        self.assertEqual(output["max_total_bytes"], 1024**3)
        self.assertEqual(output["slots"][0]["media_types"], ["video/mp4"])
        self.assertEqual(output["slots"][0]["min_files"], 1)
        self.assertEqual(output["slots"][0]["max_files"], 1)
        self.assertNotIn("disk-offload", display_of(recipe).tags)
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(
            memory,
            {
                "peak_bytes": 89_000_000_000,
                "reserve_bytes": 8_000_000_000,
            },
        )
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        self.assertEqual(
            recipe["execution"]["build"]["context"]["path"],
            "adapters/video/ltx2-pytorch",
        )
        self.assertTrue(digest and archive)
        index = _document(GENERATED / "catalog-index.json")
        entry = next(
            item
            for item in index["recipes"]
            if item["source_path"] == f"recipes/{RECIPE.name}"
        )
        self.assertEqual(entry["package"]["recipe_content_sha256"], _digest(RECIPE))


if __name__ == "__main__":
    unittest.main()
