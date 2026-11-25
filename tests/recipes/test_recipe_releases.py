from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest
from repo_root import ROOT

from recipe_tooling import catalog_index
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeRelease


@contextmanager
def isolated_root() -> Generator[Path]:
    previous = catalog_index.ROOT
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        catalog_index.ROOT = root
        try:
            yield root
        finally:
            catalog_index.ROOT = previous


class RecipeReleaseValidationTests(unittest.TestCase):
    def validate(self, document: Mapping[str, object]) -> RecipeRelease:
        recipe = json.loads(
            (
                ROOT
                / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json"
            ).read_text()
        )
        recipe["release"] = document
        return RecipeDefinition.model_validate(recipe).release

    def test_accepts_upstream_and_own_version_spellings(self) -> None:
        for version in ("1.6", "0.5.20", "4.5", "2.1.3", "0.5.15.post1"):
            with self.subTest(version=version):
                release = self.validate(
                    {"version": version, "released_at": "2026-09-17"}
                )
                assert release.version == version

    def test_rejects_invalid_dates_and_versions(self) -> None:
        for document in (
            {"version": "1.6", "released_at": "2026-02-30"},
            {"version": "1.6", "released_at": "2026-9-17"},
            {"version": "", "released_at": "2026-09-17"},
            {"version": "1.6 beta", "released_at": "2026-09-17"},
        ):
            with self.subTest(document=document), pytest.raises(ValueError):
                self.validate(document)

    def test_release_carries_no_history(self) -> None:
        with pytest.raises(ValueError):
            self.validate(
                {"version": "1.6", "released_at": "2026-09-17", "history": []}
            )


class RecipeReleaseBuildTests(unittest.TestCase):
    def test_source_bundle_rejects_files_above_hydration_limit(self) -> None:
        with isolated_root() as root:
            context = root / "adapters/demo"
            context.mkdir(parents=True)
            (context / "oversized.bin").write_bytes(b"four")
            previous = catalog_index.MAX_SOURCE_FILE_BYTES
            catalog_index.MAX_SOURCE_FILE_BYTES = 3
            try:
                with pytest.raises(
                    SystemExit,
                    match="source bundle file exceeds the Git blob hydration limit",
                ):
                    catalog_index.source_bundle(context)
            finally:
                catalog_index.MAX_SOURCE_FILE_BYTES = previous


if __name__ == "__main__":
    unittest.main()
