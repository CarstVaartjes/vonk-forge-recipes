"""The CHORD adapter's job contract: inputs, the four maps, size and colour type."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from qualification.definitions_loader import load_definitions
from spark_sweep.job_assertions import Output, evaluate

Image: Any = importlib.import_module("PIL.Image")

ROOT = Path(__file__).resolve().parents[1]
SLUG = "chord-pbr-material-pytorch-single"
RUNNER = ROOT / "adapters/image/chord-pbr-pytorch/pipelines/run.py"
FIXTURE = ROOT / "qualification/fixtures/tiny.png"
MAPS = {
    "basecolor.png": "RGB",
    "normal.png": "RGB",
    "roughness.png": "L",
    "metalness.png": "L",
}


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("chord_run", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_maps(modes: dict[str, str], size: tuple[int, int] | None = None) -> Any:
    def estimate(command: list[str], scratch: Path, deadline: float) -> None:
        staged = Path(command[command.index("--output-dir") + 1])
        staged.mkdir()
        with Image.open(command[command.index("--image") + 1]) as source:
            target = size or source.size
        for name, mode in modes.items():
            Image.new(mode, target).save(staged / name)

    return estimate


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.setattr(module, "_INPUT_ROOT", inputs)
    monkeypatch.setattr(module, "_estimate", _write_maps(MAPS))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "600")
    return module


def _stage(root: Path, name: str, content: bytes, declared: str | None = None) -> None:
    (root / name).write_bytes(content)
    entry = {"slot": "image", "name": declared or name}
    (root / "manifest.json").write_text(json.dumps({"files": [entry]}))


def _run(module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--output-mime", "image/png", "--output-dir", str(outputs)],
    )
    module.main()
    return outputs


def test_four_maps_satisfy_the_qualification(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    outputs = _run(adapter, tmp_path, monkeypatch)
    assert sorted(path.name for path in outputs.iterdir()) == sorted([*MAPS, "tmp"])
    assert list((outputs / "tmp").iterdir()) == []
    definition = load_definitions(ROOT / "qualification")["recipes"][
        f"vonk-forge/{SLUG}"
    ]
    written = [
        Output(name, "image/png", (outputs / name).read_bytes())
        for name in sorted(MAPS)
    ]
    assert evaluate(definition["assertions"], written) == {}


def test_a_wrong_colour_type_is_refused_and_nothing_is_published(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        adapter, "_estimate", _write_maps({**MAPS, "roughness.png": "RGB"})
    )
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(
        SystemExit, match="roughness.png is outside the output contract"
    ):
        _run(adapter, tmp_path, monkeypatch)
    assert sorted(path.name for path in (tmp_path / "outputs").iterdir()) == ["tmp"]


def test_a_map_of_another_size_is_refused(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_estimate", _write_maps(MAPS, size=(32, 32)))
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(SystemExit, match="outside the output contract"):
        _run(adapter, tmp_path, monkeypatch)


def test_an_undeclared_input_is_refused(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    (adapter._INPUT_ROOT / "extra.png").write_bytes(b"x")
    with pytest.raises(SystemExit, match="undeclared job input"):
        _run(adapter, tmp_path, monkeypatch)


def test_an_unsafe_name_is_refused(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(
        adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes(), declared="../tiny.png"
    )
    with pytest.raises(SystemExit, match="unsafe name"):
        _run(adapter, tmp_path, monkeypatch)


def test_a_size_over_the_limit_is_refused(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_MAX_PIXELS", 64 * 64 - 1)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(SystemExit, match="exceeds 16 megapixels"):
        _run(adapter, tmp_path, monkeypatch)
    assert not (tmp_path / "outputs").exists()


def test_an_image_file_over_the_byte_limit_is_refused(
    adapter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_MAX_IMAGE_BYTES", 10)
    _stage(adapter._INPUT_ROOT, "tiny.png", FIXTURE.read_bytes())
    with pytest.raises(SystemExit, match="1..16 MiB"):
        _run(adapter, tmp_path, monkeypatch)
