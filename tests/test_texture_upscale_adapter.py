"""The texture upscaler's job adapter: input bounds, alpha and outputs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from qualification.definitions_loader import load_definitions
from spark_sweep.job_assertions import Output, evaluate

np: Any = importlib.import_module("numpy")
Image: Any = importlib.import_module("PIL.Image")

ROOT = Path(__file__).resolve().parents[1]
SLUG = "texture-upscale-4x-nomos8ksc-pytorch-single"
RUNNER = ROOT / "adapters/image/texture-upscale-pytorch/run.py"
FIXTURE = ROOT / "qualification/fixtures/tiny.png"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("texture_upscale_run", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nearest(pixels: Any) -> Any:
    return pixels.repeat(4, axis=0).repeat(4, axis=1)


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load()
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weights")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.setattr(module, "_MODEL", weights)
    monkeypatch.setattr(module, "_INPUT_ROOT", inputs)
    monkeypatch.setattr(module, "_load_model", lambda: _nearest)
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "600")
    return module


def _stage(root: Path, name: str, content: bytes, slot: str = "image") -> None:
    (root / name).write_bytes(content)
    entry = {"slot": slot, "name": name}
    (root / "manifest.json").write_text(json.dumps({"files": [entry]}))


def _run(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Any]:
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
            "--output-dir",
            str(outputs),
        ],
    )
    module.main()
    return outputs, Image.open(outputs / "upscaled.png")


def _png(tmp_path: Path, image: Any, **options: Any) -> bytes:
    path = tmp_path / "source.png"
    image.save(path, **options)
    return path.read_bytes()


def test_rgb_texture_is_upscaled_4x_and_satisfies_the_qualification(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    outputs, image = _run(adapter, tmp_path, monkeypatch)
    assert sorted(path.name for path in outputs.iterdir()) == ["tmp", "upscaled.png"]
    assert list((outputs / "tmp").iterdir()) == []
    assert (image.mode, image.size) == ("RGB", (256, 256))
    definition = load_definitions(ROOT / "qualification")["recipes"][
        f"vonk-forge/{SLUG}"
    ]
    written = [
        Output("upscaled.png", "image/png", (outputs / "upscaled.png").read_bytes())
    ]
    assert evaluate(definition["assertions"], written) == {}


def test_transparency_is_carried_over_with_the_upscaled_size(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Image.new("RGBA", (8, 6), (10, 20, 30, 255))
    source.putpixel((3, 2), (10, 20, 30, 0))
    _stage(adapter._INPUT_ROOT, "mask.png", _png(tmp_path, source))
    _, image = _run(adapter, tmp_path, monkeypatch)
    assert (image.mode, image.size) == ("RGBA", (32, 24))
    assert image.getpixel((0, 0)) == (10, 20, 30, 255)
    assert image.getpixel((13, 9))[3] < 255


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (Image.new("RGB", (2049, 16)), "limit is 2048x2048"),
        (Image.new("RGB", (16, 2049)), "limit is 2048x2048"),
        (Image.new("I;16", (8, 8)), "I;16 images are not supported"),
    ],
)
def test_unsupported_input_is_refused_before_any_output(
    adapter: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: Any,
    message: str,
) -> None:
    _stage(adapter._INPUT_ROOT, "source.png", _png(tmp_path, image))
    with pytest.raises(SystemExit, match=message):
        _run(adapter, tmp_path, monkeypatch)
    assert not (tmp_path / "outputs").exists()


def test_input_that_is_not_png_or_jpeg_is_refused(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    webp = tmp_path / "source.webp"
    Image.new("RGB", (8, 8)).save(webp)
    _stage(adapter._INPUT_ROOT, "source.png", webp.read_bytes())
    with pytest.raises(SystemExit, match="PNG or JPEG"):
        _run(adapter, tmp_path, monkeypatch)


def test_input_must_be_exactly_one_image_slot_file(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(adapter._INPUT_ROOT, "source.png", FIXTURE.read_bytes(), slot="prompt")
    with pytest.raises(SystemExit, match="exactly one image input"):
        _run(adapter, tmp_path, monkeypatch)


def test_job_budget_stops_before_the_forward_pass(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_job_deadline", lambda: 0.0)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(SystemExit, match="declared job budget"):
        _run(adapter, tmp_path, monkeypatch)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("value", [None, "0", "3601", "soon"])
def test_job_must_declare_a_bounded_timeout(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("VONK_JOB_TIMEOUT_SECONDS")
    else:
        monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", value)
    with pytest.raises(SystemExit, match="VONK_JOB_TIMEOUT_SECONDS|between"):
        adapter._job_deadline()
