"""The BiRefNet adapter's job contract: inputs, the cutout and mask, refusals and failure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from repo_root import ROOT

from recipe_tooling.qualification.definitions_loader import load_definitions
from spark_sweep.job_assertions import Output, evaluate

Image: Any = importlib.import_module("PIL.Image")

SLUGS = ("birefnet-hr-pytorch-single", "birefnet-hr-matting-pytorch-single")
RUNNER = ROOT / "adapters/image/birefnet-pytorch/run.py"
FIXTURE = ROOT / "qualification/fixtures/tiny.png"
COLOR_TYPES = {"cutout.png": 6, "mask.png": 0}


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("birefnet_run", RUNNER)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _half_mask(image: Any) -> Any:
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (0, 0, image.width // 2, image.height))
    return mask


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load()
    model = tmp_path / "model"
    model.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.setattr(module, "_MODEL", model)
    monkeypatch.setattr(module, "_INPUT_ROOT", inputs)
    monkeypatch.setattr(module, "_load_model", lambda: _half_mask)
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "600")
    return module


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


def _run(module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    return outputs


def _png(tmp_path: Path, image: Any) -> bytes:
    path = tmp_path / "source.png"
    image.save(path)
    return path.read_bytes()


@pytest.mark.parametrize("slug", SLUGS)
def test_cutout_and_mask_satisfy_the_qualification(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slug: str
) -> None:
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch)
    assert sorted(path.name for path in outputs.iterdir()) == [
        "cutout.png",
        "mask.png",
        "tmp",
    ]
    assert list((outputs / "tmp").iterdir()) == []
    with Image.open(FIXTURE) as source:
        pixels = source.convert("RGB")
    with Image.open(outputs / "cutout.png") as cutout:
        assert (cutout.mode, cutout.size) == ("RGBA", pixels.size)
        assert cutout.getchannel("A").tobytes() == _half_mask(pixels).tobytes()
        assert cutout.convert("RGB").tobytes() == pixels.tobytes()
    with Image.open(outputs / "mask.png") as mask:
        assert (mask.mode, mask.size) == ("L", pixels.size)
    definition = load_definitions(ROOT / "qualification")["recipes"][
        f"vonk-forge/{slug}"
    ]
    written = [
        Output(name, "image/png", (outputs / name).read_bytes()) for name in COLOR_TYPES
    ]
    assert evaluate(definition["assertions"], written) == {}


def test_jpeg_input_gives_an_rgba_cutout(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (12, 8), (200, 10, 10)).save(source)
    _stage(adapter._INPUT_ROOT, "photo.jpg", source.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch)
    with Image.open(outputs / "cutout.png") as cutout:
        assert (cutout.mode, cutout.size) == ("RGBA", (12, 8))


def test_input_alpha_is_ignored_not_carried_into_the_model(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def segment(image: Any) -> Any:
        seen.append(image.mode)
        return _half_mask(image)

    monkeypatch.setattr(adapter, "_load_model", lambda: segment)
    _stage(
        adapter._INPUT_ROOT,
        "alpha.png",
        _png(tmp_path, Image.new("RGBA", (6, 4), (1, 2, 3, 0))),
    )
    _run(adapter, tmp_path, monkeypatch)
    assert seen == ["RGB"]


def _refuse_model_load() -> Any:
    raise AssertionError("the model must not load for a refused input")


def test_oversize_image_is_refused_before_the_model_loads(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_load_model", _refuse_model_load)
    _stage(adapter._INPUT_ROOT, "wide.png", _png(tmp_path, Image.new("RGB", (4097, 8))))
    with pytest.raises(SystemExit, match="limit is 4096x4096"):
        _run(adapter, tmp_path, monkeypatch)
    assert not (tmp_path / "outputs").exists()


def test_undeclared_extra_input_is_refused_before_the_model_loads(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_load_model", _refuse_model_load)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    (adapter._INPUT_ROOT / "extra.txt").write_text("not declared")
    with pytest.raises(SystemExit, match="do not match the manifest"):
        _run(adapter, tmp_path, monkeypatch)
    assert not (tmp_path / "outputs").exists()


def test_input_must_be_the_image_slot(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes(), slot="prompt")
    with pytest.raises(SystemExit, match="exactly one image input"):
        _run(adapter, tmp_path, monkeypatch)


def test_failed_generation_leaves_no_output_and_keeps_platform_scratch(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = tmp_path / "outputs"
    (outputs / "tmp").mkdir(parents=True)
    (outputs / "cache").mkdir()
    (outputs / "tmp" / "kept").write_text("x")
    (outputs / "cache" / "kept").write_text("x")

    def wrong_size(image: Any) -> Any:
        return Image.new("L", (1, 1))

    monkeypatch.setattr(adapter, "_load_model", lambda: wrong_size)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(SystemExit, match="outside the output contract"):
        _run(adapter, tmp_path, monkeypatch)
    assert sorted(path.name for path in outputs.iterdir()) == ["cache", "tmp"]
    assert (outputs / "tmp" / "kept").is_file()
    assert (outputs / "cache" / "kept").is_file()
