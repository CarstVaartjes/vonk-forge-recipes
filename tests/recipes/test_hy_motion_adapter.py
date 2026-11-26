"""HY-Motion adapter behaviour that needs no GPU, weights or the creator's code.

The BVH exporter is judged against the creator's forward kinematics (``WoodenMesh``'s
``simple_lbs`` chain, ported below), the job against refusals before any model load, the
zip against the sweep's own evaluator, and the recipe, Models and adapter against each other.
The model load and the generation are proven by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import zipfile
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from repo_root import ROOT

from recipe_tooling.qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate
from vonk_forge_contracts.recipe import RecipeJobSettings

np: Any = importlib.import_module("numpy")

ADAPTER = ROOT / "adapters/three-d/hy-motion"
SLUG = "hy-motion-1-0-pytorch-single"
FIXTURE = ROOT / "qualification/fixtures/hy-motion-prompt.txt"


def _load(name: str) -> Any:
    sys.path.insert(0, str(ADAPTER))
    try:
        spec = importlib.util.spec_from_file_location(
            f"hy_motion_{name}", ADAPTER / f"{name}.py"
        )
        assert spec
        assert spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ADAPTER))


exporter = _load("export_bvh")
sys.modules["export_bvh"] = exporter
run = _load("run")

JOINTS = exporter.JOINTS


def _skeleton(root: Path) -> Path:
    """A 52-joint tree (parent of i is (i - 1) // 2) in the creator's file layout."""
    rng = np.random.default_rng(1)
    parents = np.array([-1] + [(i - 1) // 2 for i in range(1, JOINTS)], dtype=np.int32)
    rest = rng.normal(scale=0.2, size=(JOINTS, 3)).astype(np.float32)
    directory = root / run.WOODEN
    directory.mkdir(parents=True)
    (directory / "j_template.bin").write_bytes(rest.tobytes())
    (directory / "kintree.bin").write_bytes(parents.tobytes())
    (directory / "joint_names.json").write_text(
        json.dumps([f"J{i}" for i in range(JOINTS)])
    )
    return directory


def _rotation(axis_angle: Any) -> Any:
    return exporter.rotation_matrices(np.asarray(axis_angle, dtype=np.float64))


def _creator_joints(skeleton: Any, poses: Any, trans: Any) -> Any:
    """``WoodenMesh.simple_lbs``'s chain: world = world[parent] @ [R | rest - rest[parent]]."""
    frames = poses.shape[0]
    rotations = _rotation(poses.reshape(frames, JOINTS, 3))
    relative = skeleton.rest.copy()
    relative[1:] -= skeleton.rest[list(skeleton.parents[1:])]
    world = np.zeros((frames, JOINTS, 4, 4))
    for joint in range(JOINTS):
        local = np.zeros((frames, 4, 4))
        local[:, :3, :3] = rotations[:, joint]
        local[:, :3, 3] = relative[joint]
        local[:, 3, 3] = 1.0
        parent = skeleton.parents[joint]
        world[:, joint] = local if parent < 0 else world[:, parent] @ local
    return world[:, :, :3, 3] + trans[:, None, :]


def _axis(index: int) -> Any:
    def rotate(angle: float) -> Any:
        c, s = np.cos(angle), np.sin(angle)
        rows = [[1, 0, 0], [0, c, -s], [0, s, c]]
        return np.roll(np.roll(np.array(rows), index, axis=0), index, axis=1)

    return rotate


def _bvh_joints(text: str) -> tuple[Any, list[str]]:
    """Forward kinematics of a BVH, as any importer does it: (frames, joints, 3) metres."""
    lines = text.splitlines()
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    stack: list[int] = []
    cursor = 0
    while lines[cursor] != "MOTION":
        words = lines[cursor].split()
        if words[0] in ("ROOT", "JOINT"):
            names.append(words[1])
            parents.append(stack[-1] if stack else -1)
            stack.append(len(names) - 1)
        elif words[0] == "OFFSET":
            offsets.append([float(value) for value in words[1:]])
        elif words[0] == "}":
            stack.pop()
        cursor += 1
    rows = np.array([[float(v) for v in line.split()] for line in lines[cursor + 3 :]])
    rotate_x, rotate_y, rotate_z = _axis(0), _axis(1), _axis(2)
    positions = np.zeros((rows.shape[0], len(names), 3))
    for frame, row in enumerate(rows):
        world: list[Any] = []
        for joint in range(len(names)):
            z, y, x = np.radians(row[3 + 3 * joint : 6 + 3 * joint])
            local = rotate_z(z) @ rotate_y(y) @ rotate_x(x)
            parent = parents[joint]
            if parent < 0:
                positions[frame, joint] = row[:3] / 100
                world.append(local)
            else:
                offset = np.array(offsets[joint]) / 100
                positions[frame, joint] = (
                    positions[frame, parent] + world[parent] @ offset
                )
                world.append(world[parent] @ local)
    return positions, names


def test_the_bvh_poses_the_skeleton_where_the_creators_kinematics_put_it(
    tmp_path: Path,
) -> None:
    skeleton = exporter.load_skeleton(
        _skeleton(tmp_path), tmp_path / run.WOODEN / "joint_names.json"
    )
    rng = np.random.default_rng(2)
    frames = 5
    poses = rng.normal(scale=1.4, size=(frames, JOINTS * 3))
    # the cases an Euler decomposition gets wrong: no rotation, gimbal lock either way, a half turn
    poses[0] = 0.0
    poses[1, 3:6] = [0.0, np.pi / 2, 0.0]
    poses[2, 3:6] = [0.0, -np.pi / 2, 0.0]
    poses[3, 6:9] = [np.pi, 0.0, 0.0]
    trans = rng.normal(size=(frames, 3))
    text = exporter.bvh_text(skeleton, poses, trans)
    # the same rotation reaches each pose when read as Rz Ry Rx
    probe = _axis(2)(0.4) @ _axis(1)(-0.7) @ _axis(0)(1.1)
    assert np.allclose(
        np.radians(exporter.euler_zyx_degrees(probe)), [0.4, -0.7, 1.1], atol=1e-9
    )
    positions, names = _bvh_joints(text)
    order = exporter._depth_first(exporter._children(skeleton.parents))
    assert names == [skeleton.names[i] for i in order]
    expected = _creator_joints(skeleton, poses, trans)[:, order]
    assert np.abs(positions - expected).max() < 1e-4
    assert "Frame Time: 0.0333333" in text


def test_a_skeleton_or_motion_off_the_contract_is_refused(tmp_path: Path) -> None:
    directory = _skeleton(tmp_path)
    names = directory / "joint_names.json"
    skeleton = exporter.load_skeleton(directory, names)
    poses = np.zeros((4, JOINTS * 3))
    trans = np.zeros((4, 3))
    for bad_poses, bad_trans in (
        (poses[:, :-3], trans),
        (poses, trans[:3]),
        (np.full_like(poses, np.nan), trans),
    ):
        with pytest.raises(SystemExit):
            exporter.bvh_text(skeleton, bad_poses, bad_trans)
    names.write_text(json.dumps(["twin"] * JOINTS))
    with pytest.raises(SystemExit, match="one tree"):
        exporter.load_skeleton(directory, names)


# --- recipe, models and adapter agree ----------------------------------------


def test_the_recipe_passes_exactly_the_arguments_the_adapter_parses() -> None:
    recipe = documents.recipe(SLUG)
    arguments = {item.name: item for item in recipe.runtime.arguments}
    assert set(arguments) == {"entrypoint", "output-mime", "seed", "duration-seconds"}
    assert arguments["entrypoint"].value == run.ENTRYPOINT
    assert arguments["output-mime"].value == "application/zip"
    assert isinstance(recipe.settings, RecipeJobSettings)
    knobs = recipe.settings.knobs
    assert knobs["duration-seconds"].minimum == run.MIN_DURATION_SECONDS
    assert knobs["duration-seconds"].maximum == run.MAX_DURATION_SECONDS
    assert knobs["seed"].maximum == run.MAX_SEED
    for name, knob in knobs.items():
        assert arguments[name].setting == name
        args = run._arguments(
            [
                "--entrypoint=" + run.ENTRYPOINT,
                "--output-mime=application/zip",
                "--seed=0",
                f"--duration-seconds={knob.value if name == 'duration-seconds' else 5.0}",
                "--output-dir=/outputs",
            ]
        )
        assert args.output_dir == Path("/outputs")


def test_every_file_the_adapter_reads_is_a_mounted_model_file() -> None:
    mounted: set[PurePosixPath] = set()
    for selection in documents.recipe(SLUG).models:
        paths = {
            item.id: item.path for item in documents.model(selection.model.slug).files
        }
        for item in selection.files:
            mounted.add(PurePosixPath(item.mount.target) / paths[item.file_id])
    wanted = {
        PurePosixPath(directory) / name
        for directory, names in run.MODEL_FILES.items()
        for name in names
    }
    assert wanted <= mounted


def test_the_qualification_asks_for_a_duration_the_adapter_accepts() -> None:
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    parameters = definition["parameters"]
    assert set(parameters) == set(documents.recipe(SLUG).settings.knobs)
    assert (
        run.MIN_DURATION_SECONDS
        <= parameters["duration-seconds"]
        <= run.MAX_DURATION_SECONDS
    )
    assert FIXTURE.read_text(encoding="utf-8").strip()


# --- job inputs and arguments ---------------------------------------------------


def _stage_inputs(root: Path, payload: bytes, *, name: str = "prompt.txt") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "total_bytes": len(payload),
        "files": [
            {
                "slot": "prompt",
                "name": name,
                "media_type": "text/plain",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _argv(outputs: Path, **overrides: str) -> list[str]:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "application/zip",
        "seed": "7",
        "duration-seconds": "2.0",
        "output-dir": str(outputs),
        **overrides,
    }
    return [f"--{name}={value}" for name, value in values.items()]


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "model/gltf-binary"},
        {"seed": "-1"},
        {"seed": str(run.MAX_SEED + 1)},
        {"duration-seconds": "0.5"},
        {"duration-seconds": "12.5"},
        {"duration-seconds": "nan"},
    ],
)
def test_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict[str, str]
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, **override))


@pytest.mark.parametrize(
    "payload", [b"", b"   \n", b"\xff\xfe", b"wave\x00", b"a" * 4097]
)
def test_a_prompt_outside_the_declared_slot_is_refused(
    tmp_path: Path, payload: bytes
) -> None:
    _stage_inputs(tmp_path, payload)
    with pytest.raises(SystemExit):
        run._load_prompt(tmp_path)


# --- the workspace and the generator ----------------------------------------------


def _mounts(root: Path) -> dict[str, Path]:
    assets, qwen, clip = root / "assets", root / "qwen", root / "clip"
    _skeleton(assets)
    (assets / "stats").mkdir()
    for name in ("faces", "skinIndice", "skinWeights", "v_template"):
        (assets / run.WOODEN / f"{name}.bin").write_bytes(b"x")
    qwen.mkdir()
    clip.mkdir()
    return {"assets": assets, "qwen": qwen, "clip": clip}


def test_the_workspace_lays_the_mounts_out_where_the_creators_code_reads_them(
    tmp_path: Path,
) -> None:
    mounts = _mounts(tmp_path / "mounts")
    joint_names = mounts["assets"] / run.WOODEN / "joint_names.json"
    workspace = run.build_workspace(
        tmp_path / "scratch", **mounts, joint_names=joint_names
    )
    assert (workspace / "ckpts/Qwen3-8B").resolve() == mounts["qwen"]
    assert (workspace / "ckpts/clip-vit-large-patch14").resolve() == mounts["clip"]
    assert (workspace / "stats").resolve() == mounts["assets"] / "stats"
    for name in run.WOODEN_FILES:
        assert (workspace / run.WOODEN / name).resolve() == (
            mounts["assets"] / run.WOODEN / name
        )
    assert (workspace / run.WOODEN / "joint_names.json").resolve() == joint_names


def test_the_generator_keeps_the_creators_guidance_and_reads_only_mounts() -> None:
    command = run.generation_command(
        Path("/s/prompt.txt"), duration_seconds=2.0, seed=3, output_dir=Path("/s/g")
    )
    assert command[command.index("--cfg-scale") + 1] == "5.0"
    assert command[command.index("--seed") + 1] == "3"
    assert command[command.index("--duration-seconds") + 1] == "2.0"
    assert command[command.index("--checkpoint") + 1].startswith("/models/hy-motion/")
    environment = run.generation_environment({"HOME": "/h"})
    assert environment["PYTHONPATH"] == str(run.SOURCE_ROOT)
    assert environment["USE_HF_MODELS"] == "0"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_generator_past_the_deadline_is_killed_with_its_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "child.pid"
    child = (
        "import subprocess, sys, time\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(marker)!r}, 'w').write(str(grandchild.pid))\n"
        "time.sleep(600)\n"
    )
    monkeypatch.setattr(run, "PUBLISH_RESERVE_SECONDS", 0)
    with pytest.raises(SystemExit, match="exceeded the job budget"):
        run.run_generation(
            [sys.executable, "-c", child],
            dict(os.environ),
            tmp_path,
            time.monotonic() + 3,
        )
    pid = int(marker.read_text())
    for _ in range(100):
        if not _alive(pid):
            break
        time.sleep(0.1)
    assert not _alive(pid)


# --- the job -------------------------------------------------------------------------


def _write_generated(command: list[str], *, frames: int) -> None:
    generated = Path(command[command.index("--output-dir") + 1])
    rng = np.random.default_rng(3)
    np.savez_compressed(
        generated / run.GENERATED_NPZ,
        gender=np.array(["neutral"]),
        Rh=np.zeros((frames, 3)),
        trans=rng.normal(size=(frames, 3)),
        poses=rng.normal(size=(frames, JOINTS * 3)),
        betas=np.zeros((1, 16)),
    )


def _job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, frames: int | None = None
) -> Path:
    inputs = tmp_path / "inputs"
    _stage_inputs(inputs, FIXTURE.read_bytes())
    outputs = tmp_path / "outputs"
    for name in ("tmp", "cache"):
        (outputs / name).mkdir(parents=True)
    (outputs / "cache/model.bin").write_bytes(b"kept")
    mounts = _mounts(tmp_path / "mounts")
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(run, "INPUT_ROOT", inputs)
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)
    monkeypatch.setattr(
        run,
        "build_workspace",
        partial(
            run.build_workspace,
            **mounts,
            joint_names=mounts["assets"] / run.WOODEN / "joint_names.json",
        ),
    )
    expected = run.expected_frames(2.0)

    def generate(command: list[str], *_: object) -> None:
        _write_generated(command, frames=expected if frames is None else frames)

    monkeypatch.setattr(run, "run_generation", generate)
    return outputs


def test_the_job_publishes_one_zip_the_sweep_accepts_and_no_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == [
        "cache",
        "motion.zip",
        "tmp",
    ]
    assert list((outputs / "tmp").iterdir()) == []
    assert (outputs / "cache/model.bin").read_bytes() == b"kept"
    archive = outputs / "motion.zip"
    with zipfile.ZipFile(archive) as opened:
        assert opened.namelist() == ["motion.bvh", "motion.npz"]
        assert {info.date_time for info in opened.infolist()} == {run.ZIP_TIMESTAMP}
        assert opened.read("motion.bvh").startswith(b"HIERARCHY")
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    verdict = evaluate(
        definition["assertions"],
        [Output(archive.name, "application/zip", archive.read_bytes())],
    )
    assert verdict == {}


def test_the_zip_bytes_depend_only_on_the_members(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    names = {}
    for directory, stamp in ((first, 1_000_000_000), (second, 1_500_000_000)):
        directory.mkdir()
        for member in run.MEMBERS:
            (directory / member).write_bytes(member.encode())
            os.utime(directory / member, (stamp, stamp))
        names[directory] = {member: directory / member for member in run.MEMBERS}
    one = run.publish(names[first], first, first).read_bytes()
    two = run.publish(names[second], second, second).read_bytes()
    assert one == two


def test_a_generation_off_the_contract_publishes_nothing_and_keeps_the_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path, frames=5)
    with pytest.raises(SystemExit, match="frames"):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]
    assert list((outputs / "tmp").iterdir()) == []
    monkeypatch.setattr(run, "run_generation", lambda *_: None)
    with pytest.raises(SystemExit, match="did not write"):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]
