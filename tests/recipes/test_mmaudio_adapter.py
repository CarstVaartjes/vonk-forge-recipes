"""MMAudio adapter behaviour that needs no GPU, weights or the creator's code.

The job's ingress and arguments, the worker's use of the creator's modules (faked), WAV publication
and the patch script against the recipe, Models and qualification case. The creator's model is
proven by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from repo_root import ROOT

from recipe_tooling.qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate
from vonk_forge_contracts.recipe import RecipeJobInterface, RecipeJobSettings

ADAPTER = ROOT / "adapters/audio/mmaudio-pytorch"
SLUG = "mmaudio-large-44k-v2-pytorch-single"
FIXTURES = ROOT / "qualification/fixtures"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load("mmaudio_run", ADAPTER / "run.py")
prepare = _load("mmaudio_prepare", ADAPTER / "prepare_upstream.py")


class Samples:
    """What the creator's generate returns: a tensor of shape (batch, samples)."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def float(self) -> Samples:
        return self

    def cpu(self) -> Samples:
        return self

    def __getitem__(self, index: int) -> Samples:
        return Samples(self.values[index])

    def numpy(self) -> np.ndarray:
        return self.values


class Creator:
    """Stand-ins for torch and the creator's mmaudio modules, recording what the worker hands them."""

    def __init__(self, video_seconds: float = 1.0) -> None:
        self.video_seconds = video_seconds
        self.calls: dict[str, Any] = {}
        self.config = types.SimpleNamespace(
            duration=8.0,
            sampling_rate=44100,
            latent_seq_len=345,
            clip_seq_len=64,
            sync_seq_len=192,
        )
        self.network = mock.MagicMock()
        self.network.to.return_value.eval.return_value = self.network
        self.features = mock.MagicMock()
        self.features.to.return_value.eval.return_value = self.features

    def generate(
        self, clip: Any, sync: Any, text: list[str], **keywords: Any
    ) -> Samples:
        self.calls["generate"] = (clip, sync, text, keywords, self.config.duration)
        frames = int(np.ceil(self.config.duration * 44100 / 1024)) * 1024
        return Samples(np.full((1, frames), 2.0, dtype=np.float32))

    def flow_matching(self, **keywords: Any) -> None:
        self.calls["flow"] = keywords

    def load_video(self, path: Path, duration: float, **keywords: Any) -> Any:
        self.calls["load_video"] = (path, duration, keywords)
        frames = mock.MagicMock()
        return types.SimpleNamespace(
            duration_sec=self.video_seconds, clip_frames=frames, sync_frames=frames
        )

    def modules(self) -> dict[str, types.ModuleType]:
        def module(name: str, **attributes: Any) -> types.ModuleType:
            made = types.ModuleType(name)
            made.__dict__.update(attributes)
            return made

        def get_network(variant: str) -> Any:
            self.calls["variant"] = variant
            return self.network

        def features(**keywords: Any) -> Any:
            self.calls["features"] = keywords
            return self.features

        torch = mock.MagicMock()
        # a MagicMock context manager would swallow the exception it wraps
        torch.inference_mode.return_value.__exit__.return_value = False
        return {
            "torch": torch,
            "mmaudio": module("mmaudio"),
            "mmaudio.eval_utils": module(
                "mmaudio.eval_utils", generate=self.generate, load_video=self.load_video
            ),
            "mmaudio.model": module("mmaudio.model"),
            "mmaudio.model.flow_matching": module(
                "mmaudio.model.flow_matching",
                FlowMatching=self.flow_matching,
            ),
            "mmaudio.model.networks": module(
                "mmaudio.model.networks", get_my_mmaudio=get_network
            ),
            "mmaudio.model.sequence_config": module(
                "mmaudio.model.sequence_config", CONFIG_44K=self.config
            ),
            "mmaudio.model.utils": module("mmaudio.model.utils"),
            "mmaudio.model.utils.features_utils": module(
                "mmaudio.model.utils.features_utils", FeaturesUtils=features
            ),
        }

    def worker(self) -> Any:
        with mock.patch.dict(sys.modules, self.modules()):
            return _load("mmaudio_worker", ADAPTER / "worker.py")


def _stage(root: Path, files: list[tuple[str, str, str, bytes]]) -> None:
    """Job inputs: (slot, name, media type, payload) each, with the manifest the platform writes."""
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for slot, name, media_type, payload in files:
        (root / name).write_bytes(payload)
        entries.append(
            {
                "slot": slot,
                "name": name,
                "media_type": media_type,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "total_bytes": sum(len(item[3]) for item in files),
        "files": sorted(entries, key=lambda entry: entry["name"].encode()),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


PROMPT = ("prompt", "prompt.txt", "text/plain", (FIXTURES / "prompt.txt").read_bytes())
VIDEO = (
    "video",
    "clip.mp4",
    "video/mp4",
    (FIXTURES / "foley-video-1s.mp4").read_bytes(),
)
NEGATIVE = ("negative-prompt", "negative.txt", "text/plain", b"speech, music")


def _argv(outputs: Path, **overrides: str) -> list[str]:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "audio/wav",
        "duration-seconds": "8.0",
        "seed": "42",
        "cfg-strength": "4.5",
        "num-steps": "25",
        "output-dir": str(outputs),
        **overrides,
    }
    return [f"--{name}={value}" for name, value in values.items()]


def _wav(path: Path, frames: int, *, rate: int = 44100, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\x00" * frames * channels)
    return path


# --- recipe, models and adapter agree -----------------------------------------------------------


def test_the_recipe_knobs_are_the_arguments_the_adapter_parses() -> None:
    recipe = documents.recipe(SLUG)
    assert isinstance(recipe.settings, RecipeJobSettings)
    arguments = {item.name: item for item in recipe.runtime.arguments}
    knobs = recipe.settings.knobs
    assert set(arguments) == {"entrypoint", "output-mime", "num-steps", *knobs}
    assert arguments["entrypoint"].value == run.ENTRYPOINT
    assert arguments["num-steps"].value == 25
    assert all(arguments[name].setting == name for name in knobs)
    duration, cfg = knobs["duration-seconds"], knobs["cfg-strength"]
    assert (duration.minimum, duration.maximum) == (
        run.MIN_DURATION_SECONDS,
        run.MAX_DURATION_SECONDS,
    )
    assert (cfg.minimum, cfg.maximum) == (run.MIN_CFG_STRENGTH, run.MAX_CFG_STRENGTH)
    assert knobs["seed"].maximum == run.MAX_SEED
    argv = [f"--{name}={knob.value}" for name, knob in knobs.items()]
    run._arguments(
        [
            f"--entrypoint={run.ENTRYPOINT}",
            "--output-mime=audio/wav",
            "--num-steps=25",
            *argv,
            "--output-dir=/outputs",
        ]
    )


def test_every_file_the_adapter_reads_is_a_mounted_model_file() -> None:
    mounted: set[Path] = set()
    for selection in documents.recipe(SLUG).models:
        paths = {
            item.id: item.path for item in documents.model(selection.model.slug).files
        }
        for item in selection.files:
            mounted.add(Path(item.mount.target) / paths[item.file_id])
    wanted = {
        directory / name
        for directory, names in run.MODEL_FILES.items()
        for name in names
    }
    assert wanted <= mounted


def test_the_worker_reads_the_mounted_paths_the_adapter_checks() -> None:
    worker = Creator().worker()
    read = {Path(worker.NETWORK), Path(worker.VAE), Path(worker.SYNCHFORMER)}
    assert read == {
        directory / name
        for directory, names in run.MODEL_FILES.items()
        for name in names
        if directory.name == "target"
    }


def test_the_inputs_fit_the_recipe_input_budget() -> None:
    interface = documents.recipe(SLUG).interfaces[0]
    assert isinstance(interface, RecipeJobInterface)
    declared = interface.input
    assert declared is not None
    assert declared.slots is not None
    assert sum(slot.max_total_bytes for slot in declared.slots) == declared.max_bytes


# --- job inputs -----------------------------------------------------------------------------------


def test_the_qualification_inputs_pass_the_job_ingress(tmp_path: Path) -> None:
    _stage(tmp_path, [PROMPT, VIDEO])
    inputs = run.load_inputs(tmp_path)
    assert inputs.video is not None
    assert inputs.negative_prompt is None
    _stage(tmp_path / "text", [PROMPT, NEGATIVE])
    inputs = run.load_inputs(tmp_path / "text")
    assert inputs.video is None
    assert inputs.negative_prompt is not None


@pytest.mark.parametrize(
    "files",
    [
        [VIDEO],
        [PROMPT, ("prompt", "second.txt", "text/plain", b"again")],
        [("prompt", "prompt.txt", "text/plain", b"")],
        [("prompt", "prompt.txt", "text/plain", b"\xff\xfe")],
        [("prompt", "prompt.md", "text/plain", b"sound")],
        [("prompt", "prompt.txt", "text/markdown", b"sound")],
        [PROMPT, ("negative-prompt", "n.txt", "text/plain", b"x" * 4097)],
        [PROMPT, ("video", "clip.mp4", "video/webm", VIDEO[3])],
        [PROMPT, ("video", "clip.avi", "video/mp4", VIDEO[3])],
        [PROMPT, ("image", "pic.png", "image/png", b"png")],
        [("prompt", "pro$mpt.txt", "text/plain", b"sound")],
    ],
)
def test_inputs_outside_the_declared_slots_or_off_their_type_are_refused(
    tmp_path: Path, files: list[tuple[str, str, str, bytes]]
) -> None:
    _stage(tmp_path, files)
    with pytest.raises(SystemExit):
        run.load_inputs(tmp_path)


def test_an_empty_negative_prompt_is_allowed_and_an_undeclared_file_is_not(
    tmp_path: Path,
) -> None:
    _stage(tmp_path, [PROMPT, ("negative-prompt", "negative.txt", "text/plain", b"")])
    assert run.load_inputs(tmp_path).negative_prompt is not None
    (tmp_path / "extra.txt").write_text("x")
    with pytest.raises(SystemExit, match="manifest"):
        run.load_inputs(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "audio/flac"},
        {"cfg-strength": "0.5"},
        {"cfg-strength": "11"},
        {"duration-seconds": "0.5"},
        {"duration-seconds": "31"},
        {"duration-seconds": "nan"},
        {"seed": "-1"},
        {"num-steps": "0"},
    ],
)
def test_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict[str, str]
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, **override))


def test_the_worker_command_names_each_declared_input(tmp_path: Path) -> None:
    _stage(tmp_path / "inputs", [PROMPT, VIDEO, NEGATIVE])
    inputs = run.load_inputs(tmp_path / "inputs")
    command = run.worker_command(
        inputs, run._arguments(_argv(tmp_path)), output=tmp_path / "audio.wav"
    )
    assert command[command.index("--video") + 1] == str(tmp_path / "inputs/clip.mp4")
    assert command[command.index("--negative-prompt-file") + 1].endswith("negative.txt")
    assert command[command.index("--num-steps") + 1] == "25"
    assert run.worker_environment({"HOME": "/h"})["PYTHONPATH"] == str(run.SOURCE_ROOT)


# --- the worker's use of the creator's modules ------------------------------------------------------


def _synthesize(creator: Creator, tmp_path: Path, **overrides: Any) -> np.ndarray:
    worker = creator.worker()
    arguments: dict[str, Any] = {
        "prompt": "rain",
        "negative_prompt": "",
        "video": None,
        "duration": 8.0,
        "seed": 42,
        "cfg_strength": 4.5,
        "num_steps": 25,
        **overrides,
    }
    return worker.synthesize(**arguments)


def test_with_a_video_the_videos_own_length_is_the_duration(tmp_path: Path) -> None:
    creator = Creator(video_seconds=1.0)
    samples = _synthesize(creator, tmp_path, video=tmp_path / "clip.mp4", duration=8.0)
    _, asked, keywords = creator.calls["load_video"]
    assert (asked, keywords) == (8.0, {"load_all_frames": False})
    clip, sync, text, _, duration = creator.calls["generate"]
    assert duration == 1.0
    assert clip is not None
    assert sync is not None
    assert text == ["rain"]
    # one second of the creator's 44k config: ceil(44100 / 1024) whole latents
    assert len(samples) == 45056
    creator.network.update_seq_lengths.assert_called_once_with(345, 64, 192)


def test_without_a_video_the_requested_duration_stands_and_the_negative_prompt_is_empty(
    tmp_path: Path,
) -> None:
    creator = Creator()
    _synthesize(creator, tmp_path, duration=3.0)
    clip, sync, _, keywords, duration = creator.calls["generate"]
    assert (clip, sync, duration) == (None, None, 3.0)
    assert keywords["negative_text"] == [""]
    assert keywords["cfg_strength"] == 4.5
    assert "load_video" not in creator.calls
    assert creator.calls["variant"] == "large_44k_v2"
    assert creator.calls["flow"] == {
        "min_sigma": 0,
        "inference_mode": "euler",
        "num_steps": 25,
    }
    assert creator.calls["features"]["bigvgan_vocoder_ckpt"] is None
    assert creator.calls["features"]["mode"] == "44k"


def test_samples_are_clipped_not_wrapped_into_mono_pcm16(tmp_path: Path) -> None:
    worker = Creator().worker()
    path = tmp_path / "audio.wav"
    worker.write_wav(path, np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32), 44100)
    with wave.open(str(path), "rb") as stream:
        assert (
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getframerate(),
        ) == (1, 2, 44100)
        values = np.frombuffer(stream.readframes(4), dtype="<i2").tolist()
    assert values == [0, 16384, 32767, -32767]


def test_a_non_finite_sample_is_refused_before_any_file_is_written(
    tmp_path: Path,
) -> None:
    worker = Creator().worker()
    with pytest.raises(SystemExit, match="non-finite"):
        worker.write_wav(tmp_path / "audio.wav", np.array([0.0, np.nan]), 44100)
    assert not (tmp_path / "audio.wav").exists()


# --- the job --------------------------------------------------------------------------------------


def _job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: list[tuple[str, str, str, bytes]],
    *,
    frames: int | None,
) -> Path:
    _stage(tmp_path / "inputs", files)
    outputs = tmp_path / "outputs"
    for name in ("tmp", "cache"):
        (outputs / name).mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(run, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)

    def worker(command: list[str], *_: object) -> None:
        if frames is not None:
            _wav(Path(command[command.index("--output") + 1]), frames)

    monkeypatch.setattr(run, "run_worker", worker)
    return outputs


@pytest.mark.parametrize(
    ("files", "frames"),
    [([PROMPT, VIDEO], 45056), ([PROMPT], 8 * 44100 // 1024 * 1024 + 1024)],
)
def test_the_job_publishes_one_wav_the_sweep_accepts_and_no_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: list[tuple[str, str, str, bytes]],
    frames: int,
) -> None:
    outputs = _job(monkeypatch, tmp_path, files, frames=frames)
    run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == [
        "audio.wav",
        "cache",
        "tmp",
    ]
    assert list((outputs / "tmp").iterdir()) == []
    published = outputs / "audio.wav"
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    if frames == 45056:
        verdict = evaluate(
            definition["assertions"],
            [Output(published.name, "audio/wav", published.read_bytes())],
        )
        assert verdict == {}


@pytest.mark.parametrize("frames", [None, 10, 44100 * 31])
def test_a_failed_or_off_contract_generation_publishes_nothing_and_keeps_the_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: int | None
) -> None:
    outputs = _job(monkeypatch, tmp_path, [PROMPT], frames=frames)
    with pytest.raises(SystemExit):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]


@pytest.mark.parametrize(("rate", "channels"), [(48000, 1), (44100, 2)])
def test_a_wav_of_another_rate_or_channel_count_is_refused(
    tmp_path: Path, rate: int, channels: int
) -> None:
    with pytest.raises(SystemExit, match="expected mono"):
        run.validated_wav(_wav(tmp_path / "a.wav", 2048, rate=rate, channels=channels))


# --- the patch script -----------------------------------------------------------------------------


def _source_tree(root: Path, *, clip_lines: int = 1) -> None:
    (root / "mmaudio/model/utils").mkdir(parents=True)
    (root / "mmaudio/ext/autoencoder").mkdir(parents=True)
    (root / "mmaudio/model/utils/features_utils.py").write_text(
        "create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')\n"
        * clip_lines
    )
    (root / "mmaudio/ext/autoencoder/autoencoder.py").write_text(
        "BigVGANv2.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x')\n"
    )


def test_the_patches_point_the_two_hub_fetches_at_the_mounts(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    prepare.prepare(tmp_path)
    clip = tmp_path / "mmaudio/model/utils/features_utils.py"
    vocoder = tmp_path / "mmaudio/ext/autoencoder/autoencoder.py"
    assert "local-dir:" in clip.read_text()
    assert "bigvgan" in vocoder.read_text()
    assert "hf-hub" not in clip.read_text()


@pytest.mark.parametrize("clip_lines", [0, 2])
def test_a_source_layout_that_moved_fails_the_build_instead_of_running_unpatched(
    tmp_path: Path, clip_lines: int
) -> None:
    _source_tree(tmp_path, clip_lines=clip_lines)
    with pytest.raises(SystemExit, match="unexpected MMAudio source layout"):
        prepare.prepare(tmp_path)
