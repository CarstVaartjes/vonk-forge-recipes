"""TangoFlux adapter behaviour that needs no GPU, weights or the creator's code.

The job's ingress (the one prompt, the knobs), the creator's load check and generation call
(``generation.py``, over fake model and VAE objects), the WAV check and publication, and the
recipe, Models and fixtures against each other. The creator's sampling is proven by the sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from repo_root import ROOT

from recipe_tooling.qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate
from vonk_forge_contracts.recipe import RecipeJobSettings

ADAPTER = ROOT / "adapters/audio/tangoflux-pytorch"
SLUG = "tangoflux-pytorch-single"
PROMPT = (ROOT / "qualification/fixtures/sfx-prompt.txt").read_bytes()


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Import without writing bytecode, so the tests leave no __pycache__ in the adapter directory.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


sys.path.insert(0, str(ADAPTER))
run = _load("tangoflux_run", ADAPTER / "run.py")
generation = _load("tangoflux_generation", ADAPTER / "generation.py")


def _stage(root: Path, name: str = "prompt.txt", payload: bytes = PROMPT) -> None:
    """The job inputs with the manifest the platform writes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)
    entry = {
        "slot": "prompt",
        "name": name,
        "media_type": "text/plain",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = {"schema_version": 1, "total_bytes": len(payload), "files": [entry]}
    (root / "manifest.json").write_text(json.dumps(manifest))


def _argv(outputs: Path, **overrides: str) -> list[str]:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "audio/wav",
        "duration-seconds": "2.0",
        "steps": "50",
        "seed": "4",
        "guidance-scale": "4.5",
        "output-dir": str(outputs),
        **overrides,
    }
    return [f"--{name}={value}" for name, value in values.items()]


# --- recipe, models, fixtures and adapter agree ---------------------------------------


def test_the_recipe_knobs_are_the_arguments_the_adapter_parses() -> None:
    recipe = documents.recipe(SLUG)
    assert isinstance(recipe.settings, RecipeJobSettings)
    arguments = {item.name: item for item in recipe.runtime.arguments}
    knobs = recipe.settings.knobs
    assert set(arguments) == {"entrypoint", "output-mime", "guidance-scale", *knobs}
    assert arguments["entrypoint"].value == run.ENTRYPOINT
    assert arguments["guidance-scale"].value == 4.5
    assert all(arguments[name].setting == name for name in knobs)
    duration = knobs["duration-seconds"]
    assert (duration.minimum, duration.maximum) == (run.MIN_DURATION, run.MAX_DURATION)
    steps = knobs["steps"]
    assert (steps.minimum, steps.maximum) == (run.MIN_STEPS, run.MAX_STEPS)
    assert knobs["seed"].maximum == run.MAX_SEED
    # the defaults, passed the way the platform renders them, are accepted as they stand
    argv = [f"--{name}={knob.value}" for name, knob in knobs.items()]
    args = run._arguments(
        [
            f"--entrypoint={run.ENTRYPOINT}",
            f"--output-mime={run.OUTPUT_MIME}",
            "--guidance-scale=4.5",
            *argv,
            "--output-dir=/outputs",
        ]
    )
    assert (args.duration_seconds, args.steps, args.seed) == (10.0, 50, 0)


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


def test_the_qualification_prompt_is_a_valid_input(tmp_path: Path) -> None:
    _stage(tmp_path)
    assert run.load_prompt(tmp_path) == tmp_path / "prompt.txt"


# --- job inputs and arguments ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("prompt.txt", b""),
        ("prompt.txt", b"   \n"),
        ("prompt.txt", b"\xff\xfe"),
        ("prompt.txt", b"x" * 16385),
        ("prompt.wav", b"a dog"),
    ],
)
def test_a_prompt_that_is_empty_oversize_or_not_text_is_refused(
    tmp_path: Path, name: str, payload: bytes
) -> None:
    _stage(tmp_path, name, payload)
    with pytest.raises(SystemExit):
        run.load_prompt(tmp_path)


def test_an_extra_file_or_a_wrong_slot_is_refused(tmp_path: Path) -> None:
    _stage(tmp_path / "extra")
    (tmp_path / "extra" / "other.txt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="do not match"):
        run.load_prompt(tmp_path / "extra")
    _stage(tmp_path / "slot")
    manifest = json.loads((tmp_path / "slot" / "manifest.json").read_text())
    manifest["files"][0]["slot"] = "audio"
    (tmp_path / "slot" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="prompt slot"):
        run.load_prompt(tmp_path / "slot")


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "audio/flac"},
        {"duration-seconds": "0.5"},
        {"duration-seconds": "31"},
        {"duration-seconds": "nan"},
        {"steps": "9"},
        {"steps": "101"},
        {"seed": "-1"},
        {"guidance-scale": "0"},
    ],
)
def test_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict[str, str]
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, **override))


# --- the creator's load check and generation call -------------------------------------------


def test_a_checkpoint_whose_keys_differ_from_the_creators_expectation_is_refused() -> (
    None
):
    expected = SimpleNamespace(
        missing_keys=["text_encoder.encoder.embed_tokens.weight"], unexpected_keys=[]
    )
    generation.check_load_result(expected)
    for missing, unexpected in [
        ([], []),
        (["text_encoder.encoder.embed_tokens.weight", "fc.0.weight"], []),
        (["text_encoder.encoder.embed_tokens.weight"], ["transformer.extra"]),
    ]:
        with pytest.raises(SystemExit, match="keys differ"):
            generation.check_load_result(
                SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)
            )


class _Latents:
    def transpose(self, first: int, second: int) -> _Latents:
        assert (first, second) == (2, 1)
        return self


class _Model:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def inference_flow(self, prompt: Any, **options: Any) -> _Latents:
        self.calls.append((prompt, options))
        return _Latents()


class _Vae:
    config = SimpleNamespace(sampling_rate=44100)

    def decode(self, latents: _Latents) -> Any:
        clip = np.zeros((1, 2, generation.MAX_FRAMES), dtype=np.float32)
        return SimpleNamespace(sample=SimpleNamespace(cpu=lambda: clip))


def test_the_seed_steps_duration_and_guidance_reach_the_creators_inference_flow() -> (
    None
):
    model = _Model()
    wave_out = generation.generate_wave(
        model, _Vae(), "rain", duration=2.0, steps=12, seed=9, guidance_scale=4.5
    )
    assert model.calls == [
        (
            "rain",
            {
                "duration": 2.0,
                "num_inference_steps": 12,
                "guidance_scale": 4.5,
                "seed": 9,
            },
        )
    ]
    assert wave_out.shape == (2, 88200)


def test_the_clip_length_is_the_creators_trim_capped_at_what_the_model_writes() -> None:
    assert generation.output_frames(2.0) == 88200
    assert generation.output_frames(29.95) == int(29.95 * 44100)
    assert generation.output_frames(30.0) == 645 * 2048


# --- the job -------------------------------------------------------------------------------


def _write_wav(path: Path, frames: int, channels: int = 2, rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(b"\0\0" * channels * frames)


def _job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    frames: int = 88200,
    channels: int = 2,
) -> Path:
    _stage(tmp_path / "inputs")
    outputs = tmp_path / "outputs"
    for name in ("tmp", "cache"):
        (outputs / name).mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(run, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)

    def generate(command: list[str], *_: object) -> None:
        _write_wav(Path(command[command.index("--output") + 1]), frames, channels)

    monkeypatch.setattr(run, "run_generation", generate)
    return outputs


def test_the_job_publishes_one_wav_the_sweep_accepts_and_no_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == [
        "audio.wav",
        "cache",
        "tmp",
    ]
    assert list((outputs / "tmp").iterdir()) == []
    with wave.open(str(outputs / "audio.wav"), "rb") as clip:
        assert (clip.getnchannels(), clip.getframerate(), clip.getnframes()) == (
            2,
            44100,
            88200,
        )
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    verdict = evaluate(
        definition["assertions"],
        [Output("audio.wav", "audio/wav", (outputs / "audio.wav").read_bytes())],
    )
    assert verdict == {}


@pytest.mark.parametrize(("frames", "channels"), [(88199, 2), (88200, 1)])
def test_a_clip_off_the_contract_publishes_nothing_and_keeps_the_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: int, channels: int
) -> None:
    outputs = _job(monkeypatch, tmp_path, frames=frames, channels=channels)
    with pytest.raises(SystemExit, match="not"):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]


def test_a_generation_that_wrote_nothing_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "run_generation", lambda *_: None)
    with pytest.raises(SystemExit, match="did not write"):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]
