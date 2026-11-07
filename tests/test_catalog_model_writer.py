from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools._catalog_model import add_model_arguments, write_model


def _written(tmp_path: Path, *extra: str) -> tuple[dict, dict]:
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


def test_family_publisher_is_separate_from_the_republisher(tmp_path: Path) -> None:
    model, display = _written(tmp_path, "--family-publisher", "creator")
    identity = model["identity"]
    assert identity["publisher"] == "republisher"
    assert identity["family"]["publisher"] == "creator"
    assert identity["model"]["publisher"] == "creator"
    assert display["identity"]["publisher"] == "republisher"


def test_family_publisher_defaults_to_the_publisher(tmp_path: Path) -> None:
    model, _ = _written(tmp_path)
    assert model["identity"]["family"]["publisher"] == "republisher"


def test_several_modalities_are_written_without_stray_tags(tmp_path: Path) -> None:
    model, display = _written(tmp_path, "--modality", "language", "--modality", "image")
    assert "modalities" not in model
    assert display["modalities"] == ["text", "image"]
    assert "candidate" not in display["tags"]
    assert "language" not in display["tags"]
