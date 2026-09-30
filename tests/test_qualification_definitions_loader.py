from __future__ import annotations

import json
from pathlib import Path

import pytest

from qualification.definitions_loader import load_definitions

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _root(tmp_path: Path) -> Path:
    _write(tmp_path / "shared.json", {"schema_version": 2, "fixtures": {}})
    return tmp_path


def test_assembles_sorted_maps(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root / "recipes" / "b.json", {"id": "p/b", "recipes": {"x": 1}})
    _write(root / "recipes" / "a.json", {"id": "p/a", "service_recipes": {"y": 2}})
    document = load_definitions(root)
    assert document["recipes"] == {"p/b": {"x": 1}}
    assert document["service_recipes"] == {"p/a": {"y": 2}}


def test_rejects_id_that_does_not_match_file_name(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root / "recipes" / "a.json", {"id": "p/b", "recipes": {}})
    with pytest.raises(ValueError, match="'id' must be"):
        load_definitions(root)


def test_rejects_recipe_defined_twice(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "recipes").mkdir()
    (root / "recipes" / "a.json").write_text(
        '{"id": "p/a", "recipes": {"k": 1}, "recipes": {"k": 2}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        load_definitions(root)


def test_rejects_per_recipe_map_in_shared_file(tmp_path: Path) -> None:
    _write(tmp_path / "shared.json", {"schema_version": 2, "recipes": {}})
    with pytest.raises(ValueError, match="must not hold"):
        load_definitions(tmp_path)


def test_rejects_leftover_definitions_json(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root / "definitions.json", {"schema_version": 2})
    with pytest.raises(ValueError, match="old layout"):
        load_definitions(root)


def test_repository_definitions_load() -> None:
    document = load_definitions(ROOT / "qualification")
    assert document["recipes"] and document["service_recipes"]
