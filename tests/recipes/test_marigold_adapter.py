"""The Marigold adapter's job contract: tasks, maps, colour types, alpha, refusals and seed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from repo_root import ROOT

from recipe_tooling.qualification.definitions_loader import load_definitions
from spark_sweep.job_assertions import Output, evaluate

np: Any = importlib.import_module("numpy")
Image: Any = importlib.import_module("PIL.Image")

RUNNER = ROOT / "adapters/image/marigold-diffusers/run.py"
FIXTURE = ROOT / "qualification/fixtures/tiny.png"
SLUGS = {
    "intrinsics-appearance": "marigold-iid-appearance-pytorch-single",
    "intrinsics-lighting": "marigold-iid-lighting-pytorch-single",
    "normals-depth": "marigold-normals-depth-pytorch-single",
}
# Each file's PNG (bit depth, colour type) as the adapter must write it for an RGB input.
CONTRACT = {
    "intrinsics-appearance": {
        "albedo.png": (8, 2),
        "roughness.png": (8, 0),
        "metallic.png": (8, 0),
    },
    "intrinsics-lighting": {
        "albedo.png": (8, 2),
        "shading.png": (8, 2),
        "residual.png": (8, 2),
    },
    "normals-depth": {"normal.png": (8, 2), "depth.png": (16, 0)},
}
APPEARANCE_PROPERTIES = {
    "target_names": ["albedo", "material"],
    "albedo": {"prediction_space": "srgb"},
    "material": {
        "prediction_space": "stack",
        "sub_target_names": ["roughness", "metallicity", None],
    },
}
LIGHTING_PROPERTIES = {"target_names": ["albedo", "shading", "residual"]}
ROUGHNESS, METALLICITY = 0.5, 0.25


class FakeProcessor:
    """The diffusers processor's three calls, as its source computes them."""

    @staticmethod
    def visualize_intrinsics(prediction: Any, properties: Any) -> list[Any]:
        shown = {}
        for index, name in enumerate(properties["target_names"]):
            if properties.get(name, {}).get("prediction_space") == "stack":
                # The creator colours single channels white-to-black: 0 is white.
                for channel, sub in enumerate(properties[name]["sub_target_names"]):
                    if sub:
                        gray = 255 - (prediction[index][..., channel] * 255).astype(
                            "u1"
                        )
                        shown[sub] = Image.fromarray(np.stack([gray] * 3, axis=-1))
            shown[name] = Image.fromarray((prediction[index] * 255).astype("u1"))
        return [shown]

    @staticmethod
    def visualize_normals(prediction: Any) -> list[Any]:
        return [Image.fromarray(((prediction[0] + 1.0) * 0.5 * 255).astype("u1"))]

    @staticmethod
    def export_depth_to_16bit_png(prediction: Any) -> list[Any]:
        return [Image.fromarray((prediction[0, ..., 0] * (2**16 - 1)).astype("<u2"))]


class FakePipeline:
    def __init__(self, kind: str, properties: Any, calls: list[Any]) -> None:
        self.kind = kind
        self.target_properties = properties
        self.image_processor = FakeProcessor
        self.calls = calls

    def __call__(self, image: Any, generator: Any) -> Any:
        width, height = image.size
        self.calls.append((self.kind, image.mode, generator, image.getpixel((0, 0))))
        shape = (height, width, 3)
        if self.kind == "normals":
            return SimpleNamespace(prediction=np.zeros((1, *shape), "f4"))
        if self.kind == "depth":
            return SimpleNamespace(prediction=np.full((1, height, width, 1), 0.5, "f4"))
        targets = len(self.target_properties["target_names"])
        prediction = np.full((targets, *shape), 0.2, "f4")
        if targets == 2:  # appearance: the material target holds roughness, metallic
            prediction[1, ..., 0] = ROUGHNESS
            prediction[1, ..., 1] = METALLICITY
        return SimpleNamespace(prediction=prediction)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("marigold_run", RUNNER)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    mounts = {name: tmp_path / name for name in ("target", "normals", "depth")}
    for mount in mounts.values():
        mount.mkdir()
    module.calls = []
    module.properties = APPEARANCE_PROPERTIES
    monkeypatch.setattr(module, "_INPUT_ROOT", inputs)
    for name, mount in mounts.items():
        monkeypatch.setattr(module, f"_{name.upper()}", mount)
    monkeypatch.setattr(module, "_generator", lambda seed: seed)
    monkeypatch.setattr(
        module,
        "_TASKS",
        {
            task: tuple(
                module.Stage(stage.kind, mounts[stage.mount.name], stage.render)
                for stage in stages
            )
            for task, stages in module._TASKS.items()
        },
    )
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "600")
    return module


def _use_fake(adapter: Any, monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    properties = (
        LIGHTING_PROPERTIES if task == "intrinsics-lighting" else APPEARANCE_PROPERTIES
    )
    monkeypatch.setattr(
        adapter,
        "_load_pipeline",
        lambda kind, mount: FakePipeline(kind, properties, adapter.calls),
    )


def _stage(root: Path, name: str, payload: bytes, slot: str = "image") -> None:
    (root / name).write_bytes(payload)
    media_type = "image/png" if name.endswith(".png") else "image/jpeg"
    manifest = {
        "schema_version": 1,
        "total_bytes": len(payload),
        "files": [
            {
                "slot": slot,
                "name": name,
                "media_type": media_type,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _run(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    seed: int = 7,
) -> Path:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--entrypoint",
            "/opt/vonk/source/run.py",
            "--output-mime",
            "image/png",
            "--task",
            task,
            "--seed",
            str(seed),
            "--output-dir",
            str(outputs),
        ],
    )
    module.main()
    return outputs


def _header(path: Path) -> tuple[int, int, int, int]:
    return struct.unpack(">IIBB", path.read_bytes()[16:26])


@pytest.mark.parametrize("task", sorted(SLUGS))
def test_each_task_writes_its_maps_and_satisfies_the_qualification(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    _use_fake(adapter, monkeypatch, task)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch, task)
    assert sorted(path.name for path in outputs.iterdir()) == sorted(
        [*CONTRACT[task], "tmp"]
    )
    assert list((outputs / "tmp").iterdir()) == []
    for name, (bit_depth, color_type) in CONTRACT[task].items():
        assert _header(outputs / name) == (64, 64, bit_depth, color_type)
    definition = load_definitions(ROOT / "qualification")["recipes"][
        f"vonk-forge/{SLUGS[task]}"
    ]
    written = [
        Output(name, "image/png", (outputs / name).read_bytes())
        for name in sorted(CONTRACT[task])
    ]
    assert evaluate(definition["assertions"], written) == {}


def test_roughness_and_metallic_are_the_predicted_values_not_the_inverted_display(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake(adapter, monkeypatch, "intrinsics-appearance")
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch, "intrinsics-appearance")
    with Image.open(outputs / "roughness.png") as roughness:
        assert set(np.asarray(roughness).flat) == {int(ROUGHNESS * 255)}
    with Image.open(outputs / "metallic.png") as metallic:
        assert set(np.asarray(metallic).flat) == {int(METALLICITY * 255)}


def test_depth_is_sixteen_bit_and_normals_are_rgb(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake(adapter, monkeypatch, "normals-depth")
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch, "normals-depth")
    with Image.open(outputs / "depth.png") as depth:
        assert depth.mode == "I;16"
        assert set(np.asarray(depth).flat) == {int(0.5 * (2**16 - 1))}
    assert [call[0] for call in adapter.calls] == ["normals", "depth"]


@pytest.mark.parametrize("task", ["intrinsics-appearance", "intrinsics-lighting"])
def test_input_alpha_goes_onto_the_albedo_and_the_model_sees_rgb(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    _use_fake(adapter, monkeypatch, task)
    source = Image.new("RGBA", (6, 4), (200, 100, 50, 255))
    source.putpixel((0, 0), (200, 100, 50, 0))
    path = tmp_path / "alpha.png"
    source.save(path)
    _stage(adapter._INPUT_ROOT, "alpha.png", path.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch, task)
    assert adapter.calls[0][1:2] == ("RGB",)
    assert adapter.calls[0][3] == (200, 100, 50)  # the RGB pixels, as the creator does
    assert _header(outputs / "albedo.png")[2:] == (8, 6)
    with Image.open(outputs / "albedo.png") as albedo:
        assert albedo.getchannel("A").tobytes() == source.getchannel("A").tobytes()


def test_alpha_is_not_carried_to_normals_and_depth(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake(adapter, monkeypatch, "normals-depth")
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (6, 4), (200, 100, 50, 0)).save(path)
    _stage(adapter._INPUT_ROOT, "alpha.png", path.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch, "normals-depth")
    assert _header(outputs / "normal.png")[2:] == (8, 2)


def test_the_seed_reaches_every_pipeline_generator(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake(adapter, monkeypatch, "normals-depth")
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    _run(adapter, tmp_path, monkeypatch, "normals-depth", seed=1234)
    assert [call[2] for call in adapter.calls] == [1234, 1234]


def _refuse_pipeline(kind: str, mount: Path) -> Any:
    raise AssertionError("no pipeline may load for a refused job")


def test_unknown_task_and_out_of_range_seed_are_refused_before_any_work(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_load_pipeline", _refuse_pipeline)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(SystemExit):
        _run(adapter, tmp_path, monkeypatch, "intrinsics-everything")
    with pytest.raises(SystemExit, match="seed must be"):
        _run(adapter, tmp_path, monkeypatch, "normals-depth", seed=2**31)
    assert not (tmp_path / "outputs").exists()


def test_undeclared_input_and_oversize_image_are_refused_before_any_pipeline(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_load_pipeline", _refuse_pipeline)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    (adapter._INPUT_ROOT / "extra.txt").write_text("not declared")
    with pytest.raises(SystemExit, match="do not match the manifest"):
        _run(adapter, tmp_path, monkeypatch, "intrinsics-appearance")
    (adapter._INPUT_ROOT / "extra.txt").unlink()
    wide = tmp_path / "wide.png"
    Image.new("RGB", (4097, 8)).save(wide)
    _stage(adapter._INPUT_ROOT, "wide.png", wide.read_bytes())
    (adapter._INPUT_ROOT / "tiny.png").unlink()
    with pytest.raises(SystemExit, match="limit is 4096x4096"):
        _run(adapter, tmp_path, monkeypatch, "intrinsics-appearance")
    assert not (tmp_path / "outputs").exists()


def test_a_pipeline_with_other_target_properties_fails_with_no_output(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = {"target_names": ["albedo", "shading"]}
    monkeypatch.setattr(
        adapter,
        "_load_pipeline",
        lambda kind, mount: FakePipeline(kind, other, adapter.calls),
    )
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(SystemExit, match="target properties"):
        _run(adapter, tmp_path, monkeypatch, "intrinsics-appearance")
    assert not (tmp_path / "outputs").exists()
