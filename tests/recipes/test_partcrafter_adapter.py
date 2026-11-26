"""PartCrafter adapter behaviour that needs no GPU, weights or the creator's code.

The job's ingress (the one image, the part count, the seed), the generator command, the GLB checks
and atomic publication, the source patch and the recipe, Models and fixtures against each other.
The creator's sampling is proven by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import trimesh
from repo_root import ROOT

from recipe_tooling.qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate
from vonk_forge_contracts.recipe import RecipeJobSettings

ADAPTER = ROOT / "adapters/three-d/partcrafter"
SLUG = "partcrafter-pytorch-single"
IMAGE = (ROOT / "qualification/fixtures/partcrafter-np2.png").read_bytes()


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
run = _load("partcrafter_run", ADAPTER / "run.py")
prepare = _load("partcrafter_prepare_upstream", ADAPTER / "prepare_upstream.py")


def _stage(
    root: Path,
    name: str = "object.png",
    media_type: str = "image/png",
    payload: bytes = IMAGE,
) -> None:
    """The job inputs with the manifest the platform writes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)
    entry = {
        "slot": "image",
        "name": name,
        "media_type": media_type,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = {"schema_version": 1, "total_bytes": len(payload), "files": [entry]}
    (root / "manifest.json").write_text(json.dumps(manifest))


def _argv(outputs: Path, **overrides: str) -> list[str]:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "model/gltf-binary",
        "num-parts": "2",
        "seed": "7",
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
    parts = knobs["num-parts"]
    assert (parts.minimum, parts.maximum) == (run.MIN_PARTS, run.MAX_PARTS)
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
    assert args.num_parts == 3
    assert args.seed == 0


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
    assert wanted == mounted


def test_the_qualification_image_is_the_creators_example_and_a_valid_input(
    tmp_path: Path,
) -> None:
    _stage(tmp_path)
    assert run.load_image(tmp_path) == tmp_path / "object.png"


# --- job inputs -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "media_type", "payload"),
    [
        ("object.png", "image/jpeg", IMAGE),
        ("object.png", "image/png", b"not an image"),
    ],
)
def test_an_image_that_is_not_what_the_manifest_declares_is_refused(
    tmp_path: Path, name: str, media_type: str, payload: bytes
) -> None:
    _stage(tmp_path, name, media_type, payload)
    with pytest.raises(SystemExit):
        run.load_image(tmp_path)


def test_an_extra_file_a_wrong_slot_or_a_changed_size_is_refused(
    tmp_path: Path,
) -> None:
    _stage(tmp_path)
    (tmp_path / "extra.png").write_bytes(IMAGE)
    with pytest.raises(SystemExit, match="do not match"):
        run.load_image(tmp_path)
    (tmp_path / "extra.png").unlink()
    (tmp_path / "object.png").write_bytes(IMAGE + b"\0")
    with pytest.raises(SystemExit, match="manifest"):
        run.load_image(tmp_path)
    (tmp_path / "object.png").write_bytes(IMAGE)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"][0]["slot"] = "mesh"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="image slot"):
        run.load_image(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "model/gltf+json"},
        {"num-parts": "0"},
        {"num-parts": "17"},
        {"seed": "-1"},
        {"seed": str(2**31)},
    ],
)
def test_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict[str, str]
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, **override))


# --- the generator and the source patch ---------------------------------------------------


def test_the_command_carries_the_seed_the_parts_and_the_mounted_weights(
    tmp_path: Path,
) -> None:
    args = run._arguments(_argv(tmp_path))
    command = run.generation_command(tmp_path / "o.png", args, generated=tmp_path / "g")
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--num-parts") + 1] == "2"
    assert command[command.index("--weights") + 1] == str(run.WEIGHTS)
    assert command[command.index("--rmbg") + 1] == str(run.RMBG)
    assert run.generation_environment({"HOME": "/h"})["PYTHONPATH"] == str(
        run.SOURCE_ROOT
    )


def test_the_source_patch_comments_out_the_one_import_and_refuses_other_layouts(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath(*prepare.ENCODER_FILE.parts)
    path.parent.mkdir(parents=True)
    path.write_text("import torch\nfrom torch_cluster import fps\nx = 1\n")
    prepare.main(["prepare_upstream.py", str(tmp_path)])
    patched = path.read_text()
    assert "\nfrom torch_cluster" not in patched
    assert patched.startswith("import torch\n# from torch_cluster import fps")
    with pytest.raises(SystemExit, match="unexpected PartCrafter source layout"):
        prepare.main(["prepare_upstream.py", str(tmp_path)])


# --- the job -------------------------------------------------------------------------------


def _glb(path: Path, parts: int) -> None:
    scene = trimesh.Scene()
    for index in range(parts):
        box = trimesh.creation.box(extents=(1.0 + index, 1.0, 1.0))
        box.apply_translation((3.0 * index, 0.0, 0.0))
        scene.add_geometry(box)
    scene.export(path)


def _job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    parts_written: int = 2,
    scene_parts: int = 2,
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
        generated = Path(command[command.index("--output-dir") + 1])
        for index in range(parts_written):
            _glb(generated / run.PART_NAME.format(index), 1)
        _glb(generated / run.SCENE_NAME, scene_parts)

    monkeypatch.setattr(run, "run_generation", generate)
    return outputs


def test_the_job_publishes_the_parts_and_the_scene_the_sweep_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    names = sorted(path.name for path in outputs.iterdir())
    assert names == ["cache", "object.glb", "part_00.glb", "part_01.glb", "tmp"]
    assert list((outputs / "tmp").iterdir()) == []
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    verdict = evaluate(
        definition["assertions"],
        [
            Output(name, "model/gltf-binary", (outputs / name).read_bytes())
            for name in names
            if name.endswith(".glb")
        ],
    )
    assert verdict == {}


@pytest.mark.parametrize(
    ("parts_written", "scene_parts"),
    [(1, 2), (2, 1)],
)
def test_a_missing_part_or_a_scene_of_the_wrong_size_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parts_written: int,
    scene_parts: int,
) -> None:
    outputs = _job(
        monkeypatch, tmp_path, parts_written=parts_written, scene_parts=scene_parts
    )
    with pytest.raises(SystemExit):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]


def test_a_failed_rename_removes_the_files_already_placed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(monkeypatch, tmp_path)
    real_replace = run.os.replace
    calls: list[Path] = []

    def replace(source: Path, target: Path) -> None:
        calls.append(target)
        if len(calls) == 3:
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr(run.os, "replace", replace)
    with pytest.raises(OSError, match="disk full"):
        run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]
