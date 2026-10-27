"""MotionBricks adapter behaviour that needs no GPU, weights or upstream stack.

The command contract, the job-input check, the merged ``out/`` tree the upstream
configs expect, the idle controller maths, the motion validation and the
publication all run here. The checkpoint load and the generation loop are
proven once on real weights and again by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import typing
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/three-d/motionbricks"
WHEEL = next(ADAPTER.glob("*.whl"))
FIXTURE = ROOT / "qualification/fixtures/motionbricks-command.json"
RECIPE = ROOT / "recipes/motionbricks-g1-pytorch-single.json"
MODEL = ROOT / "models/motionbricks-g1-45c03fac.json"
QUALIFICATION = ROOT / "qualification/recipes/motionbricks-g1-pytorch-single.json"
sys.path.insert(0, str(WHEEL))


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "motionbricks_run", ADAPTER / "run.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_run()


def _command(*segments: dict) -> bytes:
    return json.dumps({"schema_version": 1, "segments": list(segments)}).encode()


WALK = {"style": "walk", "duration_seconds": 2.0}


def _stage_inputs(root: Path, payload: bytes, *, slot: str = "command") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "command.json").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "total_bytes": len(payload),
        "files": [
            {
                "slot": slot,
                "name": "command.json",
                "media_type": "application/json",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


# --- command contract -------------------------------------------------------


def test_the_qualification_fixture_is_a_valid_90_frame_command() -> None:
    command = run.Command.model_validate_json(FIXTURE.read_bytes())
    assert [s.frames() for s in command.segments] == [60, 30]
    assert [s.style for s in command.segments] == ["walk", "idle"]


def test_styles_are_the_upstream_clips_without_the_blendspace_targets() -> None:
    assert typing.get_args(run.Style) == run.STYLES
    assert len(run.QPOS_LAYOUT) == run.QPOS_DIM == 36


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": 1, "segments": [{**WALK, "style": "run"}]},
        {"schema_version": 1, "segments": [{**WALK, "style": "walk_left"}]},
        {"schema_version": 1, "segments": [{**WALK, "text": "wave hello"}]},
        {"schema_version": 1, "segments": [{**WALK, "duration_seconds": 0}]},
        {"schema_version": 1, "segments": [{**WALK, "duration_seconds": -1.0}]},
        {"schema_version": 1, "segments": [{**WALK, "duration_seconds": 0.001}]},
        {"schema_version": 1, "segments": [{**WALK, "duration_seconds": 61}]},
        {"schema_version": 1, "segments": [{**WALK, "speed_mps": 0}]},
        {"schema_version": 1, "segments": [{**WALK, "speed_mps": 4.5}]},
        {"schema_version": 1, "segments": [{**WALK, "movement_heading_degrees": 361}]},
        {"schema_version": 1, "segments": []},
        {"schema_version": 1, "segments": [WALK] * 65},
        {"schema_version": 1, "segments": [{**WALK, "duration_seconds": 60}] * 3},
        {"schema_version": 2, "segments": [WALK]},
        {"segments": [WALK]},
        {"schema_version": 1, "segments": [WALK], "prompt": "dance"},
    ],
)
def test_the_command_contract_rejects(document: dict) -> None:
    with pytest.raises(ValueError):
        run.Command.model_validate_json(json.dumps(document))


def test_the_command_contract_rejects_non_finite_numbers() -> None:
    raw = '{"schema_version": 1, "segments": [{"style": "walk", "duration_seconds": NaN}]}'
    with pytest.raises(ValueError):
        run.Command.model_validate_json(raw)


def test_the_command_contract_accepts_the_full_option_set() -> None:
    command = run.Command.model_validate_json(
        _command(
            {
                "style": "walk_zombie",
                "duration_seconds": 1,
                "movement_heading_degrees": -90,
                "facing_heading_degrees": 45.5,
                "speed_mps": 1.2,
            },
            {"style": "idle", "duration_seconds": 0.5},
        )
    )
    assert [s.frames() for s in command.segments] == [30, 15]


# --- job inputs --------------------------------------------------------------


def test_the_declared_command_is_loaded(tmp_path: Path) -> None:
    _stage_inputs(tmp_path / "inputs", FIXTURE.read_bytes())
    command = run._load_command(tmp_path / "inputs")
    assert len(command.segments) == 2


def test_undeclared_inputs_are_not_silently_consumed(tmp_path: Path) -> None:
    _stage_inputs(tmp_path / "inputs", FIXTURE.read_bytes())
    (tmp_path / "inputs/extra.json").write_text("{}")
    with pytest.raises(SystemExit, match="do not match"):
        run._load_command(tmp_path / "inputs")


def test_a_changed_command_fails_its_digest(tmp_path: Path) -> None:
    _stage_inputs(tmp_path / "inputs", FIXTURE.read_bytes())
    (tmp_path / "inputs/command.json").write_bytes(
        FIXTURE.read_bytes().replace(b"walk", b"slow_walk", 1)
    )
    with pytest.raises(SystemExit, match="digest|size"):
        run._load_command(tmp_path / "inputs")


def test_a_command_in_another_slot_is_refused(tmp_path: Path) -> None:
    _stage_inputs(tmp_path / "inputs", FIXTURE.read_bytes(), slot="session")
    with pytest.raises(SystemExit, match="command"):
        run._load_command(tmp_path / "inputs")


def test_an_invalid_declared_command_fails_with_evidence(tmp_path: Path) -> None:
    _stage_inputs(tmp_path / "inputs", _command({**WALK, "style": "run"}))
    with pytest.raises(SystemExit, match="invalid motion command"):
        run._load_command(tmp_path / "inputs")


# --- the merged out/ tree the upstream hparams name ------------------------


def _fake_roots(tmp_path: Path) -> tuple[Path, Path]:
    sidecars = tmp_path / "image/out"
    for model in ("pose", "root", "vqvae"):
        base = sidecars / f"motionbricks_{model}/version_1"
        for relative in ("hparams.yaml", "skeleton/joints.p", "stats/motion/mean.npy"):
            (base / relative).parent.mkdir(parents=True, exist_ok=True)
            (base / relative).write_bytes(f"{model}:{relative}".encode())
    models = tmp_path / "models/motionbricks/out"
    for relative in run.CHECKPOINTS:
        (models / relative).parent.mkdir(parents=True, exist_ok=True)
        (models / relative).write_bytes(f"ckpt:{relative}".encode())
    return sidecars, models


def test_sidecars_and_checkpoints_resolve_from_one_cwd_relative_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecars, models = _fake_roots(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = run.build_out_tree(scratch, sidecars=sidecars, models=models)
    assert out == scratch / "out"
    monkeypatch.chdir(scratch)
    # the paths upstream reads: result_dir-relative checkpoints and configs, and
    # the cwd-relative skeleton and statistics folders named in hparams.yaml
    assert Path("out/motionbricks_pose/version_1/hparams.yaml").read_bytes() == (
        b"pose:hparams.yaml"
    )
    assert Path("out/motionbricks_pose/version_1/skeleton/joints.p").is_file()
    assert Path("out/motionbricks_vqvae/version_1/stats/motion/mean.npy").is_file()
    assert Path("out/G1-clip.ckpt").read_bytes() == b"ckpt:G1-clip.ckpt"
    for relative in run.CHECKPOINTS[1:]:
        assert Path("out", relative).read_bytes() == f"ckpt:{relative}".encode()


def test_a_missing_checkpoint_fails_fast_with_its_name(tmp_path: Path) -> None:
    sidecars, models = _fake_roots(tmp_path)
    missing = models / run.CHECKPOINTS[2]
    missing.unlink()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(SystemExit, match="not mounted") as error:
        run.build_out_tree(scratch, sidecars=sidecars, models=models)
    assert str(missing) in str(error.value)


def test_every_path_the_vendored_configs_name_is_in_the_merged_tree(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models/motionbricks/out"
    for relative in run.CHECKPOINTS:
        (models / relative).parent.mkdir(parents=True, exist_ok=True)
        (models / relative).write_bytes(b"x")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = run.build_out_tree(
        scratch, sidecars=ADAPTER / "motionbricks/out", models=models
    )
    named = set()
    for hparams in (ADAPTER / "motionbricks/out").glob("*/version_1/hparams.yaml"):
        named.update(re.findall(r"\bout/[A-Za-z0-9_./=-]+", hparams.read_text()))
    assert named
    for path in sorted(named):
        target = scratch / path.rstrip("/")
        assert target.exists(), f"{path} named by {hparams.parent} is not in the tree"
    assert (out / "G1-clip.ckpt").is_file()


def test_the_recipe_mounts_the_checkpoints_where_the_adapter_reads_them() -> None:
    recipe = json.loads(RECIPE.read_text())
    model = json.loads(MODEL.read_text())
    paths = {file["id"]: file["path"] for file in model["files"]}
    (selection,) = recipe["models"]
    mounted = {
        PurePosixPath(item["mount"]["target"]) / paths[item["file_id"]]
        for item in selection["files"]
    }
    assert mounted == {run.MODEL_ROOT / relative for relative in run.CHECKPOINTS}
    assert model["source"]["revision"] == run.SOURCE_COMMIT


def test_the_vendored_sidecars_are_exactly_the_non_lfs_files_of_the_model_tree() -> (
    None
):
    vendored = {
        path.relative_to(ADAPTER / "motionbricks/out").as_posix()
        for path in (ADAPTER / "motionbricks/out").rglob("*")
        if path.is_file()
    }
    assert not vendored & set(run.CHECKPOINTS)
    assert len(vendored) == 24


# --- idle controller ---------------------------------------------------------


def test_idle_keeps_the_momentum_and_facing() -> None:
    moving = [(0.01 * step, 0.0) for step in range(5)]
    movement, facing = run._idle_directions(moving, (1.0, 0.0, 0.0, 0.0))
    assert movement == pytest.approx((1.0, 0.0), abs=1e-3)
    assert facing == pytest.approx((1.0, 0.0), abs=1e-3)
    half = math.sqrt(0.5)
    _, turned = run._idle_directions(moving, (half, 0.0, 0.0, half))  # yaw +90 deg
    assert turned == pytest.approx((0.0, 1.0), abs=1e-3)


def test_idle_standing_still_has_no_movement_direction() -> None:
    movement, _ = run._idle_directions([(1.0, 2.0)] * 5, (1.0, 0.0, 0.0, 0.0))
    assert movement == (0.0, 0.0)


# --- validation and the published document -------------------------------------


def _rows(frames: int) -> list[list[float]]:
    return [[0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0, *([0.0] * 29)] for _ in range(frames)]


def test_a_valid_trajectory_passes() -> None:
    run.validate_qpos(_rows(3), 3)


@pytest.mark.parametrize("defect", ["count", "width", "nan", "quaternion"])
def test_an_invalid_trajectory_is_refused(defect: str) -> None:
    rows = _rows(3)
    if defect == "width":
        rows[1] = rows[1][:-1]
    elif defect == "nan":
        rows[2][10] = float("nan")
    elif defect == "quaternion":
        rows[0][3] = 1.5
    expected = 4 if defect == "count" else 3
    with pytest.raises(SystemExit):
        run.validate_qpos(rows, expected)


def test_the_document_satisfies_the_qualification_assertions() -> None:
    command = run.Command.model_validate_json(FIXTURE.read_bytes())
    document = run.motion_document(command, _rows(90), seed=0)
    text = json.dumps(document, allow_nan=False)
    assert len(text) >= 1024
    assertion = next(
        item
        for item in json.loads(QUALIFICATION.read_text())["recipes"]["assertions"]
        if item["kind"] == "json-document"
    )
    assert set(assertion["required_keys"]) <= set(document)
    for key, value in assertion["equals"].items():
        assert document[key] == value
    assert [s["start_frame"] for s in document["segments"]] == [0, 60]
    assert document["segments"][1]["facing_heading_degrees"] == 0.0
    assert document["source"]["commit"] == run.SOURCE_COMMIT


# --- publication ---------------------------------------------------------------


class _Agent:
    def reset(self) -> None:
        pass


def _main_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **patches) -> Path:
    inputs = tmp_path / "inputs"
    _stage_inputs(inputs, FIXTURE.read_bytes())
    sidecars, models = _fake_roots(tmp_path)
    outputs = tmp_path / "outputs"
    (outputs / "tmp").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    tempfile.tempdir = None
    monkeypatch.setattr(run, "INPUT_ROOT", inputs)
    monkeypatch.setattr(run, "SIDECAR_ROOT", sidecars)
    monkeypatch.setattr(run, "MODEL_ROOT", models)
    monkeypatch.setattr(run, "SOURCE_ROOT", tmp_path / "source")
    monkeypatch.setattr(run, "load_agent", lambda out, **_: _Agent())
    monkeypatch.setattr(run, "seed_everything", lambda seed: None)
    monkeypatch.setattr(run, "generate_qpos", lambda *_: _rows(90))
    for name, value in patches.items():
        monkeypatch.setattr(run.os if name == "replace" else run, name, value)
    return outputs


def _argv(outputs: Path) -> list[str]:
    return [
        "--entrypoint", run.ENTRYPOINT,
        "--output-mime", "application/json",
        "--timeout-seconds", "60",
        "--seed", "7",
        "--output-dir", str(outputs),
    ]  # fmt: skip


def test_the_job_publishes_one_plain_file_and_leaves_no_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = Path.cwd()
    outputs = _main_with(monkeypatch, tmp_path)
    try:
        run.main(_argv(outputs))
        assert Path.cwd() == before
        assert tempfile.tempdir is None
    finally:
        os.chdir(before)
    assert sorted(path.name for path in outputs.iterdir()) == ["motion.json", "tmp"]
    assert list((outputs / "tmp").iterdir()) == []
    document = json.loads((outputs / "motion.json").read_text())
    assert document["frame_count"] == 90 and document["seed"] == 7


def test_a_failed_publication_leaves_no_partial_output_and_a_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = Path.cwd()
    calls = []

    def broken(source, target):
        calls.append((Path(source).name, Path(target).name))
        raise OSError("injected rename failure")

    outputs = _main_with(monkeypatch, tmp_path, replace=broken)
    try:
        with pytest.raises(OSError, match="injected"):
            run.main(_argv(outputs))
        assert Path.cwd() == before and tempfile.tempdir is None
        assert calls == [("motion.json", "motion.json")]
        assert [path.name for path in outputs.iterdir()] == ["tmp"]
        assert list((outputs / "tmp").iterdir()) == []
        monkeypatch.undo()
        outputs = _main_with(monkeypatch, tmp_path)
        run.main(_argv(outputs))
    finally:
        os.chdir(before)
    assert (outputs / "motion.json").is_file()


def test_an_invalid_generation_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = Path.cwd()
    outputs = _main_with(monkeypatch, tmp_path, generate_qpos=lambda *_: _rows(89))
    try:
        with pytest.raises(SystemExit, match="expected 90"):
            run.main(_argv(outputs))
    finally:
        os.chdir(before)
        tempfile.tempdir = None
    assert [path.name for path in outputs.iterdir()] == ["tmp"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--entrypoint", "/other.py"],
        ["--output-mime", "model/gltf-binary"],
        ["--timeout-seconds", "0"],
        ["--seed", "-1"],
    ],
)
def test_job_arguments_outside_the_harness_contract_are_refused(
    tmp_path: Path, extra: list[str]
) -> None:
    argv = _argv(tmp_path)
    argv[argv.index(extra[0]) + 1] = extra[1]
    with pytest.raises(SystemExit):
        run._arguments(argv)
