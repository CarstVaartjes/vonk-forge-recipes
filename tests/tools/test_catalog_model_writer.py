from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from recipe_tooling.catalog_model import add_model_arguments, roles_for, write_model


def _written(tmp_path: Path, *extra: str) -> tuple[dict[str, Any], dict[str, Any]]:
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    args = parser.parse_args(
        [
            "--root", str(tmp_path),
            "--publisher", "republisher",
            "--model-group-slug", "lineage",
            "--model-group-title", "Lineage",
            "--model-group-description", "A lineage.",
            "--model-slug", "lineage-model",
            "--model-title", "Lineage model",
            "--model-description", "A model.",
            "--version-slug", "lineage-bf16",
            "--version", "bf16",
            "--precision", "bf16",
            "--quantization", "none",
            "--license-spdx", "MIT",
            "--license-url", "https://example.com/license",
            "--attribution", "Example",
            *extra,
        ]
    )  # fmt: skip
    files = [
        {
            "id": "weights",
            "path": "model.safetensors",
            "roles": ["weights"],
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
    ]
    source = {"repository": "https://example.com/o/m", "revision": "a" * 40}
    # The written text is the point: the typed readers would default a missing kind.
    texts: list[str] = []

    def record(path: Path, text: str, **_: object) -> int:
        texts.append(text)  # the Model is written first, then its display
        return len(text)

    with patch.object(Path, "write_text", autospec=True, side_effect=record):
        write_model(args, source=source, files=files, requires_token=False)
    first, second = texts
    return json.loads(first), json.loads(second)


def test_display_carries_its_kind(tmp_path: Path) -> None:
    _, display = _written(tmp_path)
    assert display["kind"] == "model-display"


def test_model_publisher_is_separate_from_the_republisher(tmp_path: Path) -> None:
    model, display = _written(tmp_path, "--model-publisher", "creator")
    identity = model["identity"]
    assert identity["publisher"] == "republisher"
    assert identity["model"]["publisher"] == "creator"
    assert display["identity"]["publisher"] == "republisher"


def test_model_publisher_defaults_to_the_publisher(tmp_path: Path) -> None:
    model, _ = _written(tmp_path)
    assert model["identity"]["model"]["publisher"] == "republisher"


def test_the_hierarchy_is_written_to_the_display_only(tmp_path: Path) -> None:
    lineage = {
        "family": "lineage",
        "line": "lineage-line",
        "model": "lineage-model",
        "variant_kind": "quantization",
        "base_model": "creator/Lineage-Model",
    }
    model, display = _written(tmp_path, "--lineage", json.dumps(lineage))
    assert "family" not in model["identity"]
    assert display["lineage"] == lineage


def test_a_helper_has_no_lineage(tmp_path: Path) -> None:
    _, display = _written(tmp_path)
    assert "lineage" not in display


def test_several_modalities_are_written_without_stray_tags(tmp_path: Path) -> None:
    model, display = _written(tmp_path, "--modality", "language", "--modality", "image")
    assert "modalities" not in model
    assert display["modalities"] == ["text", "image"]
    assert "candidate" not in display["tags"]
    assert "language" not in display["tags"]


def test_every_weight_format_is_classified_as_weights() -> None:

    for name in (
        "m.safetensors",
        "m.gguf",
        "m.pth",
        "m.pt",
        "m.bin",
        "m.onnx",
        "m.ckpt",
    ):
        assert roles_for(f"sub/{name}") == ["weights"], name
    assert roles_for("notes.txt") == ["metadata"]
