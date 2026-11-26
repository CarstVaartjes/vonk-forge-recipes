"""Hunyuan3D 2.1 adapter behaviour that needs no GPU, weights or the creator's code.

The GLB writer (the creator's material mapping, trimesh's image alignment), the job's ingress and
modes, publication and the patch script against the recipes, Models and qualification cases. The
creator's shape and paint steps are proven by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from repo_root import ROOT

from recipe_tooling.qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate
from spark_sweep.job_media import glb_structure
from vonk_forge_contracts.recipe import RecipeJobSettings

ADAPTER = ROOT / "adapters/three-d/hunyuan3d-2-1"
FULL = "hunyuan3d-2-1-pytorch-single"
PAINT = "hunyuan3d-2-1-paint-pytorch-single"
FIXTURES = ROOT / "qualification/fixtures"
QUAD = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"


def _load(name: str, path: Path) -> Any:
    sys.path.insert(0, str(ADAPTER))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export = _load("hunyuan3d_export", ADAPTER / "export_pbr_glb.py")
run = _load("hunyuan3d_run", ADAPTER / "run.py")
prepare = _load("hunyuan3d_prepare", ADAPTER / "prepare_upstream.py")


def _painted_glb(root: Path, size: int = 8) -> bytes:
    """The paint step's OBJ and three JPEG maps, written to one GLB."""
    (root / "textured.obj").write_text(QUAD)
    rng = np.random.default_rng(size)
    for name in ("textured", "textured_metallic", "textured_roughness"):
        pixels = rng.integers(0, 255, (size, size + 3, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / f"{name}.jpg")
    output = root / "mesh.glb"
    export.write_pbr_glb(
        root / "textured.obj",
        root / "textured.jpg",
        root / "textured_metallic.jpg",
        root / "textured_roughness.jpg",
        output,
    )
    return output.read_bytes()


def _chunks(glb: bytes) -> tuple[dict[str, Any], bytes]:
    length = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20 : 20 + length]), glb[28 + length :]


# --- the GLB writer ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [5, 8, 13])
def test_the_painted_textures_become_a_pbr_glb_both_validators_accept(
    tmp_path: Path, size: int
) -> None:
    glb = _painted_glb(tmp_path, size)
    (tmp_path / "check.glb").write_bytes(glb)
    sys.modules["glb_validation"].validate_mesh_glb(
        tmp_path / "check.glb", profile="textured-pbr"
    )
    glb_structure(glb, {"profile": "textured-pbr"})


def test_the_metallic_roughness_image_is_the_creators_channel_mapping(
    tmp_path: Path,
) -> None:
    metallic = Image.fromarray(np.full((2, 2), 30, dtype=np.uint8))
    roughness = Image.fromarray(np.full((2, 2), 200, dtype=np.uint8))
    pixel = export.metallic_roughness_image(metallic, roughness).getpixel((0, 0))
    assert pixel == (255, 200, 30)
    document, _ = _chunks(_painted_glb(tmp_path))
    material = document["materials"][0]["pbrMetallicRoughness"]
    assert (
        material["baseColorTexture"]["index"]
        != material["metallicRoughnessTexture"]["index"]
    )


def test_a_mesh_without_uv_coordinates_is_refused(tmp_path: Path) -> None:
    (tmp_path / "bare.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n")
    Image.new("RGB", (4, 4)).save(tmp_path / "a.jpg")
    with pytest.raises(ValueError, match="UV"):
        export.write_pbr_glb(
            tmp_path / "bare.obj",
            tmp_path / "a.jpg",
            tmp_path / "a.jpg",
            tmp_path / "a.jpg",
            tmp_path / "out.glb",
        )
    assert not (tmp_path / "out.glb").exists()


# --- recipes, Models and adapter agree ----------------------------------------------------------


def test_the_recipes_pass_their_modes_and_the_full_one_its_seed() -> None:
    for slug, mode, knobs in ((FULL, "full", {"seed"}), (PAINT, "paint", set())):
        recipe = documents.recipe(slug)
        assert isinstance(recipe.settings, RecipeJobSettings)
        assert set(recipe.settings.knobs) == knobs
        arguments = {item.name: item for item in recipe.runtime.arguments}
        assert set(arguments) == {"entrypoint", "output-mime", "mode", *knobs}
        assert arguments["entrypoint"].value == run.ENTRYPOINT
        assert arguments["mode"].value == mode
    settings = documents.recipe(FULL).settings
    assert isinstance(settings, RecipeJobSettings)
    seed = settings.knobs["seed"]
    assert (seed.minimum, seed.maximum) == (0, run.MAX_SEED)
    run._arguments(
        [
            f"--entrypoint={run.ENTRYPOINT}",
            "--output-mime=model/gltf-binary",
            "--mode=full",
            f"--seed={seed.value}",
            "--output-dir=/outputs",
        ]
    )


@pytest.mark.parametrize(("slug", "mode"), [(FULL, "full"), (PAINT, "paint")])
def test_every_file_the_adapter_reads_is_a_mounted_model_file(
    slug: str, mode: str
) -> None:
    mounted: set[Path] = set()
    for selection in documents.recipe(slug).models:
        paths = {
            item.id: item.path for item in documents.model(selection.model.slug).files
        }
        for item in selection.files:
            mounted.add(Path(item.mount.target) / paths[item.file_id])
    files = (
        run.PAINT_FILES if mode == "paint" else {**run.SHAPE_FILES, **run.PAINT_FILES}
    )
    wanted = {directory / name for directory, names in files.items() for name in names}
    assert wanted <= mounted
    assert (run.DIT / "model.fp16.ckpt" in mounted) == (mode == "full")


# --- job inputs -----------------------------------------------------------------------------------


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


IMAGE = (
    "image",
    "demo.png",
    "image/png",
    (FIXTURES / "hunyuan3d-2-1-demo.png").read_bytes(),
)
MESH = ("mesh", "tiny.glb", "model/gltf-binary", (FIXTURES / "tiny.glb").read_bytes())


def _argv(outputs: Path, kind: str, **overrides: str | None) -> list[str]:
    values: dict[str, str | None] = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "model/gltf-binary",
        "mode": kind,
        "seed": "1234" if kind == "full" else None,
        "output-dir": str(outputs),
        **overrides,
    }
    return [f"--{name}={value}" for name, value in values.items() if value is not None]


def test_the_qualification_inputs_pass_the_job_ingress(tmp_path: Path) -> None:
    _stage(tmp_path / "full", [IMAGE])
    assert run.load_inputs(tmp_path / "full", "full").mesh is None
    _stage(tmp_path / "paint", [IMAGE, MESH])
    inputs = run.load_inputs(tmp_path / "paint", "paint")
    assert inputs.mesh is not None
    assert inputs.mesh.name == "tiny.glb"


@pytest.mark.parametrize(
    ("mode", "files"),
    [
        ("full", [MESH]),
        ("full", [IMAGE, MESH]),
        ("paint", [IMAGE]),
        ("paint", [MESH]),
        ("full", [("image", "demo.png", "image/png", b"not a png")]),
        ("full", [("image", "demo.jpg", "image/jpeg", IMAGE[3])]),
        ("full", [("image", "demo.png", "image/jpeg", IMAGE[3])]),
        ("paint", [IMAGE, ("mesh", "tiny.glb", "model/gltf-binary", b"glTF")]),
        ("paint", [IMAGE, ("mesh", "empty.obj", "model/obj", b"# nothing\n")]),
        ("full", [("image", "demo$.png", "image/png", IMAGE[3])]),
    ],
)
def test_inputs_outside_the_declared_slots_or_off_their_type_are_refused(
    tmp_path: Path, mode: str, files: list[tuple[str, str, str, bytes]]
) -> None:
    _stage(tmp_path, files)
    with pytest.raises(SystemExit):
        run.load_inputs(tmp_path, mode)


def test_an_extra_or_changed_input_file_is_refused(tmp_path: Path) -> None:
    _stage(tmp_path, [IMAGE])
    (tmp_path / "extra.txt").write_text("x")
    with pytest.raises(SystemExit, match="manifest"):
        run.load_inputs(tmp_path, "full")
    (tmp_path / "extra.txt").unlink()
    (tmp_path / "demo.png").write_bytes(IMAGE[3] + b"\0")
    with pytest.raises(SystemExit, match="manifest"):
        run.load_inputs(tmp_path, "full")


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "image/png"},
        {"mode": "texture"},
        {"seed": None},
        {"seed": "-1"},
        {"seed": "10000001"},
    ],
)
def test_full_mode_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict[str, str | None]
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, "full", **override))


def test_paint_mode_takes_no_seed(tmp_path: Path) -> None:
    run._arguments(_argv(tmp_path, "paint"))
    with pytest.raises(SystemExit, match="seed"):
        run._arguments(_argv(tmp_path, "paint", seed="0"))


def test_the_worker_gets_the_mode_a_scratch_mesh_copy_and_the_source_trees(
    tmp_path: Path,
) -> None:
    _stage(tmp_path / "inputs", [IMAGE, MESH])
    inputs = run.load_inputs(tmp_path / "inputs", "paint")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    command = run.worker_command(
        inputs, run._arguments(_argv(tmp_path, "paint")), scratch=scratch
    )
    assert command[command.index("--mode") + 1] == "paint"
    assert command[command.index("--mesh") + 1] == str(scratch / "input.glb")
    assert (scratch / "input.glb").read_bytes() == MESH[3]
    assert "--seed" not in command
    assert run.worker_environment({"HOME": "/h"})["PYTHONPATH"].split(":") == [
        str(path) for path in run.SOURCE_PATHS
    ]


# --- the job --------------------------------------------------------------------------------------


def _job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, *, glb: bytes | None
) -> Path:
    _stage(tmp_path / "inputs", [IMAGE] if mode == "full" else [IMAGE, MESH])
    outputs = tmp_path / "outputs"
    for name in ("tmp", "cache"):
        (outputs / name).mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(run, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)

    def worker(command: list[str], *_: object) -> None:
        if glb is not None:
            Path(command[command.index("--output") + 1]).write_bytes(glb)

    monkeypatch.setattr(run, "run_worker", worker)
    return outputs


@pytest.mark.parametrize(("slug", "mode"), [(FULL, "full"), (PAINT, "paint")])
def test_the_job_publishes_one_pbr_glb_the_sweep_accepts_and_no_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, slug: str, mode: str
) -> None:
    painted = tmp_path / "painted"
    painted.mkdir()
    outputs = _job(monkeypatch, tmp_path, mode, glb=_painted_glb(painted))
    run.main(_argv(outputs, mode))
    assert sorted(path.name for path in outputs.iterdir()) == [
        "cache",
        "mesh.glb",
        "tmp",
    ]
    assert list((outputs / "tmp").iterdir()) == []
    published = outputs / "mesh.glb"
    definition = Definitions.load().artifact(f"vonk-forge/{slug}")
    assert definition is not None
    verdict = evaluate(
        definition["assertions"],
        [Output(published.name, "model/gltf-binary", published.read_bytes())],
    )
    assert verdict == {}


@pytest.mark.parametrize("glb", [None, b"glTF not a glb"])
def test_a_failed_or_invalid_paint_publishes_nothing_and_keeps_the_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, glb: bytes | None
) -> None:
    outputs = _job(monkeypatch, tmp_path, "paint", glb=glb)
    with pytest.raises(SystemExit):
        run.main(_argv(outputs, "paint"))
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]


def test_the_geometry_only_glb_is_not_a_pbr_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _job(
        monkeypatch, tmp_path, "full", glb=(FIXTURES / "tiny.glb").read_bytes()
    )
    with pytest.raises(SystemExit, match="invalid PBR GLB"):
        run.main(_argv(outputs, "full"))
    assert not (outputs / "mesh.glb").exists()


# --- the patch script -----------------------------------------------------------------------------


def _source_tree(root: Path, *, bpy_lines: int = 1) -> None:
    (root / "hy3dpaint/DifferentiableRenderer").mkdir(parents=True)
    (root / "hy3dpaint/DifferentiableRenderer/mesh_utils.py").write_text(
        "import os\n" + "import bpy\n" * bpy_lines + "import cv2\n"
    )
    (root / "hy3dpaint/utils").mkdir(parents=True)
    (root / "hy3dpaint/utils/multiview_utils.py").write_text(
        "        model_path = huggingface_hub.snapshot_download(\n"
        "            repo_id=config.multiview_pretrained_path,\n"
        '            allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],\n'
        "        )\n"
    )


def test_the_patches_remove_bpy_and_the_download(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    prepare.prepare(tmp_path)
    assert (
        "bpy"
        not in (tmp_path / "hy3dpaint/DifferentiableRenderer/mesh_utils.py").read_text()
    )
    assert (tmp_path / "hy3dpaint/utils/multiview_utils.py").read_text() == (
        "        model_path = config.multiview_pretrained_path\n"
    )


def test_a_source_layout_that_moved_fails_the_build_instead_of_running_unpatched(
    tmp_path: Path,
) -> None:
    _source_tree(tmp_path, bpy_lines=2)
    with pytest.raises(SystemExit, match="unexpected Hunyuan3D 2.1 source layout"):
        prepare.prepare(tmp_path)
    _source_tree(tmp_path / "other", bpy_lines=0)
    with pytest.raises(SystemExit, match="unexpected Hunyuan3D 2.1 source layout"):
        prepare.prepare(tmp_path / "other")
