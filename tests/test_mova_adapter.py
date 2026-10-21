from __future__ import annotations

import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/video/mova-pytorch"
RECIPES = (
    ROOT / "recipes/mova-360p-diffusers-single.json",
    ROOT / "recipes/mova-720p-diffusers-single.json",
)


def load_runner():
    path = ADAPTER / "run.py"
    module = types.ModuleType("mova_runner")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def write_manifest(root: Path, slot: str, path: Path) -> None:
    payload = path.read_bytes()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_bytes": len(payload),
                "files": [
                    {
                        "slot": slot,
                        "name": path.name,
                        "media_type": "text/plain",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class MovaInputContractTests(unittest.TestCase):
    def test_legacy_prompt_has_no_hidden_default(self) -> None:
        runner = load_runner()
        previous = os.environ.pop("VONK_PROMPT", None)
        try:
            with self.assertRaisesRegex(SystemExit, "between 1 and 4096"):
                runner._prompt(None)
        finally:
            if previous is not None:
                os.environ["VONK_PROMPT"] = previous

    def test_manifest_prompt_is_verified_and_loaded(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.txt"
            prompt.write_text(
                "A quiet lake with synchronized water sounds", encoding="utf-8"
            )
            write_manifest(root, "prompt", prompt)
            runner._INPUT_ROOT = root
            manifest = runner._input_manifest()
            self.assertIsNotNone(manifest)
            self.assertEqual(
                runner._prompt(manifest),
                "A quiet lake with synchronized water sounds",
            )
            self.assertEqual(runner._slot_files(manifest, "reference-image"), [])

    def test_manifest_digest_mismatch_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.txt"
            prompt.write_text("Valid prompt", encoding="utf-8")
            write_manifest(root, "prompt", prompt)
            document = json.loads((root / "manifest.json").read_text())
            document["files"][0]["sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
            runner._INPUT_ROOT = root
            with self.assertRaisesRegex(SystemExit, "digest"):
                runner._input_manifest()

    def test_output_requires_exact_h264_aac_stream_contract(self) -> None:
        runner = load_runner()
        probe = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "8.041667",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 352,
                    "avg_frame_rate": "24/1",
                    "nb_read_frames": "193",
                    "duration": "8.041667",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "duration": "8.041667",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".mp4") as media:
            Path(media.name).write_bytes(b"0" * 1024)
            with mock.patch.object(
                runner.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=json.dumps(probe)),
            ) as run:
                runner._verify_synchronized_mp4(Path(media.name), height=352, width=640)
            self.assertIn("-count_frames", run.call_args.args[0])

            for stream, codec in ((0, "vp9"), (1, "opus")):
                invalid = json.loads(json.dumps(probe))
                invalid["streams"][stream]["codec_name"] = codec
                with (
                    mock.patch.object(
                        runner.subprocess,
                        "run",
                        return_value=SimpleNamespace(stdout=json.dumps(invalid)),
                    ),
                    self.assertRaisesRegex(SystemExit, "H.264 video and AAC audio"),
                ):
                    runner._verify_synchronized_mp4(
                        Path(media.name), height=352, width=640
                    )

    def test_recipes_declare_prompt_optional_reference_and_one_mp4(self) -> None:
        for path in RECIPES:
            with self.subTest(recipe=path.name):
                recipe = json.loads(path.read_text(encoding="utf-8"))
                interface = recipe["interfaces"][0]
                slots = {slot["id"]: slot for slot in interface["input"]["slots"]}
                self.assertEqual(
                    (slots["prompt"]["min_files"], slots["prompt"]["max_files"]), (1, 1)
                )
                self.assertEqual(slots["prompt"]["media_types"], ["text/plain"])
                self.assertEqual(
                    (
                        slots["reference-image"]["min_files"],
                        slots["reference-image"]["max_files"],
                    ),
                    (0, 1),
                )
                output = interface["output"]
                self.assertEqual(output["max_total_bytes"], 536870912)
                self.assertEqual(output["slots"][0]["media_types"], ["video/mp4"])
                self.assertIn("H.264", output["slots"][0]["description"])
                self.assertIn("AAC", output["slots"][0]["description"])
                self.assertEqual(
                    (
                        output["slots"][0]["min_files"],
                        output["slots"][0]["max_files"],
                    ),
                    (1, 1),
                )


if __name__ == "__main__":
    unittest.main()


def test_minimal_job_reaches_official_runner_with_native_resolution(tmp_path: Path):
    """The reviewed case must survive argv, generation and output validation."""
    import sys

    import pytest

    for recipe_path, dimensions in zip(RECIPES, ((352, 640), (720, 1280)), strict=True):
        runner = load_runner()
        spec = json.loads(
            (ROOT / "qualification/recipes" / recipe_path.name).read_text()
        )["recipes"]
        args = [
            "mova",
            "--entrypoint",
            runner._ENTRYPOINT,
            "--output-mime",
            "video/mp4",
            "--output-dir",
            str(tmp_path),
        ]
        for name, value in spec["parameters"].items():
            args.extend((f"--{name}", str(value)))
        generated: list[list[str]] = []

        def generate(
            command,
            environment,
            deadline,
            generated=generated,
            runner=runner,
            spec=spec,
        ):
            generated.append(command)
            assert environment["HF_HUB_OFFLINE"] == "1"
            assert 0 < runner._remaining(deadline) <= spec["timeout_seconds"]
            Path(command[command.index("--output_path") + 1]).write_bytes(b"mp4")

        with (
            mock.patch.dict(
                os.environ, {"VONK_JOB_TIMEOUT_SECONDS": str(spec["timeout_seconds"])}
            ),
            mock.patch.object(sys, "argv", args),
            mock.patch.object(runner, "_input_manifest", return_value=None),
            mock.patch.object(runner, "_prompt", return_value="A small sample"),
            mock.patch.object(runner, "_variant", return_value=dimensions),
            mock.patch.object(runner, "_OFFICIAL_RUNNER", Path(__file__)),
            mock.patch.object(
                runner, "_reference_frame", return_value=tmp_path / "reference.png"
            ),
            mock.patch.object(runner, "_generate", side_effect=generate),
            mock.patch.object(runner, "_verify_synchronized_mp4") as verify,
        ):
            runner.main()
        command = generated[0]
        assert command[command.index("--offload") + 1] == "none"
        assert "--remove_video_dit" not in command
        for flag, value in (
            ("--num_frames", "1"),
            ("--num_inference_steps", "2"),
            ("--height", str(dimensions[0])),
            ("--width", str(dimensions[1])),
        ):
            assert command[command.index(flag) + 1] == value
        assert verify.call_args.kwargs["frames"] == 1
        assert 0 < verify.call_args.kwargs["timeout"] <= spec["timeout_seconds"]
        assert (tmp_path / "mova.mp4").read_bytes() == b"mp4"
        with (
            mock.patch.object(sys, "argv", [*args, "--num-frames", "2"]),
            pytest.raises(SystemExit, match="4k\\+1"),
        ):
            runner._arguments()


def test_job_budget_expiry_kills_generation_with_evidence(tmp_path: Path):
    """Inject a stuck real subprocess; it must leave no artifact or live worker."""
    import signal
    import sys
    import time

    import pytest

    runner = load_runner()
    artifact = tmp_path / "late.mp4"
    script = "import time,pathlib; time.sleep(10); pathlib.Path(__import__('sys').argv[1]).write_text('late')"
    started = time.monotonic()
    with (
        mock.patch.dict(os.environ, {"VONK_JOB_TIMEOUT_SECONDS": "1"}),
        mock.patch.object(runner.os, "killpg", wraps=os.killpg) as kill,
        pytest.raises(SystemExit, match="declared job budget"),
    ):
        runner._generate(
            [sys.executable, "-c", script, str(artifact)],
            dict(os.environ),
            runner._job_deadline(),
        )
    assert time.monotonic() - started < 3
    assert kill.call_args.args[1] == signal.SIGKILL
    assert not artifact.exists()


def test_missing_or_invalid_job_budget_fails_before_generation():
    import pytest

    runner = load_runner()
    for value in (None, "", "0", "3601", "forever"):
        with mock.patch.dict(os.environ, {}, clear=True):
            if value is not None:
                os.environ["VONK_JOB_TIMEOUT_SECONDS"] = value
            with pytest.raises(SystemExit, match="job.*(timeout|TIMEOUT)"):
                runner._job_deadline()
