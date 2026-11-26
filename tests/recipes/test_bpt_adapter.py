"""BPT adapter behaviour that needs no GPU, weights or the creator's code.

The job's ingress (the one mesh, the temperature, the seed), the generator command and workspace,
the exact scale restoration, the GLB check and publication, and the recipe, Models and fixtures
against each other. The creator's sampling is proven by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from repo_root import ROOT

from recipe_tooling.qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate
from vonk_forge_contracts.recipe import RecipeJobSettings

ADAPTER = ROOT / "adapters/three-d/bpt"
SLUG = "bpt-pytorch-single"
FIXTURE = (ROOT / "qualification/fixtures/tiny.glb").read_bytes()
# The AABB of the unit cube scaled by ten and moved by (5, -3, 2): minimum corner, then maximum.
EXPECTED_BOUNDS = [0.0, -8.0, -3.0, 10.0, 2.0, 7.0]
MEDIA_TYPES = {
    ".glb": "model/gltf-binary",
    ".obj": "model/obj",
    ".ply": "model/ply",
}


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
run = _load("bpt_run", ADAPTER / "run.py")


def _stage(root: Path, name: str, payload: bytes, slot: str = "mesh") -> None:
    """The job inputs with the manifest the platform writes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)
    entry = {
        "slot": slot,
        "name": name,
        "media_type": MEDIA_TYPES[Path(name).suffix],
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = {"schema_version": 1, "total_bytes": len(payload), "files": [entry]}
    (root / "manifest.json").write_text(json.dumps(manifest))


def _argv(outputs: Path, **overrides: str) -> list[str]:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "model/gltf-binary",
        "temperature": "0.5",
        "seed": "3",
        "output-dir": str(outputs),
        **overrides,
    }
    return [f"--{name}={value}" for name, value in values.items()]


def _moved_cube(suffix: str) -> bytes:
    """A unit cube scaled by ten and moved off the origin: the scale to restore."""
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    box.apply_translation((5.0, -3.0, 2.0))
    exported: Any = box.export(file_type=suffix.lstrip("."))
    return exported if isinstance(exported, bytes) else exported.encode()


# --- recipe, models, fixtures and adapter agree ---------------------------------------


def test_the_recipe_knobs_are_the_arguments_the_adapter_parses() -> None:
    recipe = documents.recipe(SLUG)
    assert isinstance(recipe.settings, RecipeJobSettings)
    arguments = {item.name: item for item in recipe.runtime.arguments}
    knobs = recipe.settings.knobs
    assert set(arguments) == {"entrypoint", "output-mime", *knobs}
    assert arguments["entrypoint"].value == run.ENTRYPOINT
    assert all(arguments[name].setting == name for name in knobs)
    temperature = knobs["temperature"]
    assert (temperature.minimum, temperature.maximum) == (
        run.MIN_TEMPERATURE,
        run.MAX_TEMPERATURE,
    )
    assert knobs["seed"].maximum == run.MAX_SEED
    # the defaults, passed the way the platform renders them, are accepted as they stand
    argv = [f"--{name}={knob.value}" for name, knob in knobs.items()]
    args = run._arguments(
        [
            f"--entrypoint={run.ENTRYPOINT}",
            f"--output-mime={run.OUTPUT_MIME}",
            *argv,
            "--output-dir=/outputs",
        ]
    )
    assert args.temperature == 0.5
    assert args.seed == 0


def test_the_checkpoint_the_adapter_reads_is_a_mounted_model_file() -> None:
    mounted: set[Path] = set()
    for selection in documents.recipe(SLUG).models:
        paths = {
            item.id: item.path for item in documents.model(selection.model.slug).files
        }
        for item in selection.files:
            mounted.add(Path(item.mount.target) / paths[item.file_id])
    assert mounted == {run.CHECKPOINT}


def test_the_qualification_fixture_is_a_valid_input(tmp_path: Path) -> None:
    _stage(tmp_path, "tiny.glb", FIXTURE)
    source = run.load_mesh(tmp_path)
    assert source.faces == 12
    assert source.longest_side == pytest.approx(1.0)


# --- job inputs and arguments ---------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".glb", ".obj", ".ply"])
def test_each_declared_mesh_format_loads_with_its_own_bounds(
    tmp_path: Path, suffix: str
) -> None:
    _stage(tmp_path, f"cube{suffix}", _moved_cube(suffix))
    source = run.load_mesh(tmp_path)
    assert source.bounds.ravel().tolist() == pytest.approx(EXPECTED_BOUNDS)
    assert source.center.tolist() == pytest.approx([5.0, -3.0, 2.0])
    assert source.longest_side == pytest.approx(10.0)


def test_a_mesh_without_faces_or_an_undeclared_file_is_refused(tmp_path: Path) -> None:
    cloud = trimesh.PointCloud(np.random.default_rng(0).random((50, 3)))
    payload: Any = cloud.export(file_type="ply")
    _stage(tmp_path / "cloud", "cloud.ply", payload)
    with pytest.raises(SystemExit, match="not a triangle mesh|could not be loaded"):
        run.load_mesh(tmp_path / "cloud")
    _stage(tmp_path / "extra", "tiny.glb", FIXTURE)
    (tmp_path / "extra" / "other.glb").write_bytes(FIXTURE)
    with pytest.raises(SystemExit, match="do not match"):
        run.load_mesh(tmp_path / "extra")


def test_a_wrong_slot_a_changed_size_or_a_mismatched_type_is_refused(
    tmp_path: Path,
) -> None:
    _stage(tmp_path / "slot", "tiny.glb", FIXTURE, slot="image")
    with pytest.raises(SystemExit, match="mesh slot"):
        run.load_mesh(tmp_path / "slot")
    _stage(tmp_path / "size", "tiny.glb", FIXTURE)
    (tmp_path / "size" / "tiny.glb").write_bytes(FIXTURE + b"\0")
    with pytest.raises(SystemExit, match="manifest"):
        run.load_mesh(tmp_path / "size")
    _stage(tmp_path / "type", "tiny.glb", FIXTURE)
    manifest = json.loads((tmp_path / "type" / "manifest.json").read_text())
    manifest["files"][0]["media_type"] = "model/obj"
    (tmp_path / "type" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="media type"):
        run.load_mesh(tmp_path / "type")


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "model/obj"},
        {"temperature": "0.05"},
        {"temperature": "1.5"},
        {"temperature": "nan"},
        {"seed": "-1"},
        {"seed": str(2**31)},
    ],
)
def test_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict[str, str]
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, **override))


# --- the workspace, the generator and the scale ---------------------------------------------


def test_the_workspace_links_the_creators_miche_config_and_the_command_carries_the_seed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bpt"
    (source / "miche").mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    workspace = run.build_workspace(root, source=source)
    assert (workspace / "miche").resolve() == source / "miche"
    _stage(tmp_path / "in", "tiny.glb", FIXTURE)
    mesh = run.load_mesh(tmp_path / "in")
    args = run._arguments(_argv(tmp_path))
    command = run.generation_command(mesh, args, output=tmp_path / "o.npz")
    assert command[command.index("--seed") + 1] == "3"
    assert command[command.index("--temperature") + 1] == "0.5"
    assert command[command.index("--checkpoint") + 1] == str(run.CHECKPOINT)
    assert run.generation_environment({"HOME": "/h"})["PYTHONPATH"] == str(
        run.SOURCE_ROOT
    )


# --- the job -------------------------------------------------------------------------------


def _job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, written: bool = True
) -> Path:
    _stage(tmp_path / "inputs", "cube.glb", _moved_cube(".glb"))
    outputs = tmp_path / "outputs"
    for name in ("tmp", "cache"):
        (outputs / name).mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(run, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)

    def generate(command: list[str], *_: object) -> None:
        if not written:
            return
        # The model's mesh in the creator's normalised cube: a box over [-0.95, 0.95].
        box = trimesh.creation.box(extents=(1.9, 1.9, 1.9))
        np.savez(
            Path(command[command.index("--output") + 1]),
            vertices=box.vertices,
            faces=box.faces,
        )

    monkeypatch.setattr(run, "run_generation", generate)
    return outputs


def test_the_job_restores_the_sources_bounding_box_and_the_sweep_accepts_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == [
        "cache",
        "retopo.glb",
        "tmp",
    ]
    assert list((outputs / "tmp").iterdir()) == []
    produced = trimesh.load(outputs / "retopo.glb", force="mesh")
    assert produced.bounds.ravel().tolist() == pytest.approx(EXPECTED_BOUNDS, abs=1e-5)
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    verdict = evaluate(
        definition["assertions"],
        [
            Output(
                "retopo.glb", "model/gltf-binary", (outputs / "retopo.glb").read_bytes()
            )
        ],
    )
    assert verdict == {}


def test_a_generation_that_wrote_no_mesh_publishes_nothing_and_keeps_the_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path, written=False)
    with pytest.raises(SystemExit, match="did not write"):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]


def test_an_empty_generated_mesh_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)

    def generate(command: list[str], *_: object) -> None:
        np.savez(
            Path(command[command.index("--output") + 1]),
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
        )

    monkeypatch.setattr(run, "run_generation", generate)
    with pytest.raises(SystemExit, match="empty mesh"):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]
