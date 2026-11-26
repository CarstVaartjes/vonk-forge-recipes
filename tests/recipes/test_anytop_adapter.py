"""AnyTop adapter behaviour that needs no GPU, weights or the creator's code.

The job's ingress (the rig's clips, the optional rest pose, the face joints), the generator
command, the BVH retiming, the deterministic zip and the recipe, Models and fixtures against each
other. The creator's preprocessing and sampling are proven by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
from repo_root import ROOT

from recipe_tooling.qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate
from vonk_forge_contracts.recipe import RecipeJobSettings

ADAPTER = ROOT / "adapters/three-d/anytop"
SLUG = "anytop-all-pytorch-single"
FIXTURES = ROOT / "qualification/fixtures"
CLIP_A = (FIXTURES / "anytop-rig-walk-a.bvh").read_bytes()
CLIP_B = (FIXTURES / "anytop-rig-walk-b.bvh").read_bytes()
REST = (FIXTURES / "anytop-rig-tpos.bvh").read_bytes()


def _load_run() -> Any:
    spec = importlib.util.spec_from_file_location("anytop_run", ADAPTER / "run.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_run()


def _stage(root: Path, files: list[tuple[str, str, bytes]]) -> None:
    """Job inputs: (slot, name, payload) each, with the manifest the platform writes."""
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for slot, name, payload in files:
        (root / name).write_bytes(payload)
        entries.append(
            {
                "slot": slot,
                "name": name,
                "media_type": "text/plain",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "total_bytes": sum(len(payload) for _, _, payload in files),
        "files": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


QUALIFICATION_FILES = [
    ("clips", "walk-a.bvh", CLIP_A),
    ("clips", "walk-b.bvh", CLIP_B),
    ("rest-pose", "tpos.bvh", REST),
]


def _argv(outputs: Path, **overrides: str) -> list[str]:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "application/zip",
        "seed": "10",
        "motion-length-seconds": "2.0",
        "num-repetitions": "2",
        "face-joint-right-hip": "RLeg1",
        "face-joint-left-hip": "LLeg1",
        "face-joint-right-shoulder": "RArm1",
        "face-joint-left-shoulder": "LArm1",
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
    assert set(arguments) == {"entrypoint", "output-mime", *knobs}
    assert arguments["entrypoint"].value == run.ENTRYPOINT
    assert all(arguments[name].setting == name for name in knobs)
    length = knobs["motion-length-seconds"]
    assert (length.minimum, length.maximum) == (
        run.MIN_MOTION_SECONDS,
        run.MAX_MOTION_SECONDS,
    )
    repetitions = knobs["num-repetitions"]
    assert (repetitions.minimum, repetitions.maximum) == (
        run.MIN_REPETITIONS,
        run.MAX_REPETITIONS,
    )
    assert knobs["seed"].maximum == run.MAX_SEED
    # the defaults, passed the way the platform renders them, are accepted as they stand
    argv = [f"--{name}={knob.value}" for name, knob in knobs.items()]
    args = run._arguments(
        [
            f"--entrypoint={run.ENTRYPOINT}",
            "--output-mime=application/zip",
            *argv,
            "--output-dir=/outputs",
        ]
    )
    assert args.face_joints == ["RLeg1", "LLeg1", "RArm1", "LArm1"]


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


def test_the_fixtures_are_the_generators_bytes_and_pass_the_job_inputs(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate-qualification-bvh-rig"),
            str(tmp_path),
        ],
        check=True,
    )
    for name in ("walk-a", "walk-b", "tpos"):
        committed = (FIXTURES / f"anytop-rig-{name}.bvh").read_bytes()
        assert (tmp_path / f"anytop-rig-{name}.bvh").read_bytes() == committed
    _stage(tmp_path / "inputs", QUALIFICATION_FILES)
    inputs = run.load_inputs(tmp_path / "inputs")
    assert [path.name for path in inputs.clips] == ["walk-a.bvh", "walk-b.bvh"]
    assert inputs.rest_pose is not None
    assert inputs.rest_pose.name == "tpos.bvh"


# --- job inputs -------------------------------------------------------------------------


def test_a_rig_given_by_one_clip_needs_a_rest_pose(tmp_path: Path) -> None:
    _stage(tmp_path / "bare", [("clips", "walk-a.bvh", CLIP_A)])
    with pytest.raises(SystemExit, match="at least two clips"):
        run.load_inputs(tmp_path / "bare")
    _stage(
        tmp_path / "posed",
        [("clips", "walk-a.bvh", CLIP_A), ("rest-pose", "tpos.bvh", REST)],
    )
    assert run.load_inputs(tmp_path / "posed").rest_pose is not None


@pytest.mark.parametrize(
    "files",
    [
        [("clips", "walk-a.bvh", b"HIERARCHY only")],
        [("clips", "walk-a.bvh", b"\xff\xfe")],
        [("clips", "walk-a.bvh", CLIP_A), ("clips", "WALK-A.bvh", CLIP_B)],
        [("clips", "walk-a.txt", CLIP_A), ("clips", "walk-b.bvh", CLIP_B)],
        [("clips", "walk$a.bvh", CLIP_A), ("clips", "walk-b.bvh", CLIP_B)],
        [("poses", "walk-a.bvh", CLIP_A), ("clips", "walk-b.bvh", CLIP_B)],
        [
            ("clips", "walk-a.bvh", CLIP_A),
            ("rest-pose", "tpos.bvh", REST),
            ("rest-pose", "tpos2.bvh", REST),
        ],
    ],
)
def test_inputs_outside_the_declared_slots_are_refused(
    tmp_path: Path, files: list[tuple[str, str, bytes]]
) -> None:
    _stage(tmp_path, files)
    with pytest.raises(SystemExit):
        run.load_inputs(tmp_path)


def test_a_clip_whose_size_differs_from_the_manifest_or_an_extra_file_is_refused(
    tmp_path: Path,
) -> None:
    _stage(tmp_path, QUALIFICATION_FILES)
    (tmp_path / "extra.bvh").write_bytes(CLIP_A)
    with pytest.raises(SystemExit, match="do not match"):
        run.load_inputs(tmp_path)
    (tmp_path / "extra.bvh").unlink()
    (tmp_path / "walk-a.bvh").write_bytes(CLIP_A + b"\n")
    with pytest.raises(SystemExit, match="manifest"):
        run.load_inputs(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "model/gltf-binary"},
        {"seed": "-1"},
        {"motion-length-seconds": "0.5"},
        {"motion-length-seconds": "10"},
        {"motion-length-seconds": "nan"},
        {"num-repetitions": "0"},
        {"num-repetitions": "9"},
        {"face-joint-left-hip": "bad name"},
        {"face-joint-left-hip": ""},
    ],
)
def test_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict[str, str]
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, **override))


# --- the workspace and the generator -------------------------------------------------------


def test_the_workspace_and_command_hand_the_creator_the_rig_and_the_rest_pose(
    tmp_path: Path,
) -> None:
    _stage(tmp_path / "inputs", QUALIFICATION_FILES)
    inputs = run.load_inputs(tmp_path / "inputs")
    mount = tmp_path / "t5"
    mount.mkdir()
    root = tmp_path / "scratch"
    root.mkdir()
    workspace = run.build_workspace(root, inputs, t5=mount)
    assert (workspace / "t5-base").resolve() == mount
    assert sorted(path.name for path in (root / "bvhs").iterdir()) == [
        "tpos.bvh",
        "walk-a.bvh",
        "walk-b.bvh",
    ]
    args = run._arguments(_argv(tmp_path))
    command = run.generation_command(inputs, args, root=root)
    assert command[command.index("--rest-pose") + 1] == str(root / "bvhs/tpos.bvh")
    assert command[
        command.index("--face-joints") + 1 : command.index("--face-joints") + 5
    ] == [
        "RLeg1",
        "LLeg1",
        "RArm1",
        "LArm1",
    ]
    assert command[command.index("--seed") + 1] == "10"
    assert command[command.index("--num-repetitions") + 1] == "2"
    assert run.generation_environment({"HOME": "/h"})["PYTHONPATH"] == (
        f"{run.SOURCE_ROOT}:{run.MOTION_ROOT}"
    )


def test_the_creators_bvh_header_gets_the_frame_time_of_its_20_fps() -> None:
    text = "HIERARCHY\nMOTION\nFrames: 40\nFrame Time: 0.041667\n0 0\n"
    assert "Frame Time: 0.050000\n" in run.retimed(text)
    assert run.retimed(text).count("Frame Time") == 1


# --- the job -------------------------------------------------------------------------------


def _motion(frames: int) -> str:
    rows = [
        " ".join(
            f"{(joint * 7919 + frame * 104729) % 10007 / 97:.4f}" for joint in range(54)
        )
        for frame in range(frames)
    ]
    header = f"HIERARCHY\nROOT Hips\n{{\n}}\nMOTION\nFrames: {frames}\nFrame Time: 0.041667\n"
    return header + "\n".join(rows) + "\n"


def _job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    frames: int = 40,
    written: int = 2,
) -> Path:
    _stage(tmp_path / "inputs", QUALIFICATION_FILES)
    outputs = tmp_path / "outputs"
    for name in ("tmp", "cache"):
        (outputs / name).mkdir(parents=True)
    mount = tmp_path / "t5"
    mount.mkdir()
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(run, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)
    original = run.build_workspace
    monkeypatch.setattr(
        run, "build_workspace", lambda root, inputs: original(root, inputs, t5=mount)
    )

    def generate(command: list[str], *_: object) -> None:
        generated = Path(command[command.index("--output-dir") + 1])
        for index in range(written):
            (generated / run.GENERATED.format(index)).write_text(_motion(frames))

    monkeypatch.setattr(run, "run_generation", generate)
    return outputs


def test_the_job_publishes_one_zip_the_sweep_accepts_and_no_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == [
        "cache",
        "motions.zip",
        "tmp",
    ]
    assert list((outputs / "tmp").iterdir()) == []
    archive = outputs / "motions.zip"
    with zipfile.ZipFile(archive) as opened:
        assert opened.namelist() == ["motion-000.bvh"]
        assert {info.date_time for info in opened.infolist()} == {run.ZIP_TIMESTAMP}
        assert b"Frame Time: 0.050000" in opened.read("motion-000.bvh")
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    verdict = evaluate(
        definition["assertions"],
        [Output(archive.name, "application/zip", archive.read_bytes())],
    )
    assert verdict == {}


@pytest.mark.parametrize(("frames", "written"), [(7, 2), (40, 1)])
def test_a_generation_off_the_contract_publishes_nothing_and_keeps_the_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: int, written: int
) -> None:
    outputs = _job(monkeypatch, tmp_path, frames=frames, written=written)
    with pytest.raises(SystemExit):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]
    assert list((outputs / "tmp").iterdir()) == []
