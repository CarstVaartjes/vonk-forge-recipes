from __future__ import annotations

import json
from pathlib import Path

import pytest
from generated_catalog import GENERATED


@pytest.fixture(scope="session")
def built_catalog() -> tuple[dict, Path]:
    """The catalog and package directory built once for the whole session.

    One build reads, hashes and gzips the closure of all recipes, which is the
    most expensive thing this suite does. Every test reads the same generated
    outputs; the cases that exercise the archive validators consume the
    produced bytes without writing to the package directory.
    """
    catalog = json.loads((GENERATED / "catalog-index.json").read_text(encoding="utf-8"))
    return catalog, GENERATED / "packages"
