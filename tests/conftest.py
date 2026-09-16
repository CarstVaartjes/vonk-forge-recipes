from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CATALOG_TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))


@pytest.fixture(scope="session")
def built_catalog(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    """Build every recipe package once, then share the result for the session.

    One build reads, hashes and gzips the closure of all recipes, which is the
    most expensive thing this suite does. The returned catalog does not depend
    on where the archives are written, and the cases that exercise the archive
    validators consume the produced bytes without writing to the package
    directory, so regenerating the catalog per test buys nothing.
    """
    package_dir = tmp_path_factory.mktemp("recipe-packages")
    return CATALOG_TOOL["build"](package_dir=package_dir), package_dir
