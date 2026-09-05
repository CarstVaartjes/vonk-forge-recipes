from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import RecipeDefinition  # noqa: E402
from vonk_forge_contracts.recipe import RecipeRelease  # noqa: E402
SCRIPT = ROOT / "tools/build-catalog-index"
LOADER = importlib.machinery.SourceFileLoader("build_catalog_index", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
catalog_index = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = catalog_index
LOADER.exec_module(catalog_index)

OLDER_DIGEST = "b" * 64


def release_document() -> dict[str, object]:
    return {
        "version": "2.0.0",
        "released_at": "2026-08-23",
        "history": [
            {
                "version": "2.0.0",
                "released_at": "2026-08-23",
                "prior_recipe_content_sha256": None,
                "upgrade_effect": "rebuild",
                "changes": [
                    {
                        "kind": "runtime",
                        "summary": "Refresh the runtime integration.",
                        "details": "Rebuild the image before installing this release.",
                        "references": ["https://example.com/releases/2.0.0"],
                    }
                ],
            },
            {
                "version": "1.0.0",
                "released_at": "2026-08-01",
                "prior_recipe_content_sha256": OLDER_DIGEST,
                "upgrade_effect": "reprepare",
                "changes": [
                    {
                        "kind": "initial",
                        "summary": "Initial reviewed catalog release.",
                    }
                ],
            },
        ],
    }


@contextmanager
def isolated_root() -> Iterator[Path]:
    previous = catalog_index.ROOT
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        catalog_index.ROOT = root
        try:
            yield root
        finally:
            catalog_index.ROOT = previous


class RecipeReleaseValidationTests(unittest.TestCase):
    def validate(self, document: dict[str, object]) -> RecipeRelease:
        recipe = json.loads(
            (ROOT / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json").read_text()
        )
        recipe["release"] = document
        parsed_recipe = RecipeDefinition.model_validate(recipe)
        self.assertEqual(parsed_recipe.identity.publisher, "vonk-forge")
        self.assertEqual(parsed_recipe.identity.slug, "synthetic-tiny-build")
        release = parsed_recipe.release
        return release

    def assert_invalid(self, document: dict[str, object]) -> None:
        with self.assertRaises(ValueError):
            self.validate(document)

    def test_accepts_current_release_without_self_digest(self) -> None:
        release = self.validate(release_document())

        self.assertEqual(release.version, "2.0.0")
        self.assertEqual(release.released_at, "2026-08-23")
        history = release.history
        self.assertEqual(
            [item.prior_recipe_content_sha256 for item in history],
            [None, OLDER_DIGEST],
        )

    def test_rejects_invalid_historical_digest(self) -> None:
        document = release_document()
        document["history"][1]["prior_recipe_content_sha256"] = "invalid"

        self.assert_invalid(document)

    def test_rejects_legacy_sidecar_identity_field(self) -> None:
        document = release_document()
        document["recipe"] = {"publisher": "example", "slug": "demo"}

        self.assert_invalid(document)

    def test_rejects_unsorted_or_duplicate_history(self) -> None:
        cases: dict[str, dict[str, object]] = {}

        duplicate_version = release_document()
        duplicate_version["history"][1]["version"] = "2.0.0"
        cases["duplicate version"] = duplicate_version

        unsorted_date = release_document()
        unsorted_date["history"][1]["released_at"] = "2026-08-24"
        cases["unsorted date"] = unsorted_date

        for name, document in cases.items():
            with self.subTest(name=name):
                self.assert_invalid(document)

    def test_rejects_invalid_date_category_and_upgrade_effect(self) -> None:
        cases: dict[str, dict[str, object]] = {}

        invalid_date = release_document()
        invalid_date["released_at"] = "2026-02-30"
        cases["date"] = invalid_date

        invalid_category = release_document()
        invalid_category["history"][0]["changes"][0]["kind"] = "marketing"
        cases["category"] = invalid_category

        invalid_effect = release_document()
        invalid_effect["history"][0]["upgrade_effect"] = "redeploy"
        cases["upgrade effect"] = invalid_effect

        for name, document in cases.items():
            with self.subTest(name=name):
                self.assert_invalid(document)

    def test_rejects_bounded_text_and_collection_overflows(self) -> None:
        cases: dict[str, dict[str, object]] = {}

        long_summary = release_document()
        long_summary["history"][0]["changes"][0]["summary"] = "s" * 161
        cases["summary"] = long_summary

        long_details = release_document()
        long_details["history"][0]["changes"][0]["details"] = "d" * 1001
        cases["details"] = long_details

        too_many_references = release_document()
        too_many_references["history"][0]["changes"][0]["references"] = [
            f"https://example.com/{index}" for index in range(9)
        ]
        cases["references"] = too_many_references

        empty_changes = release_document()
        empty_changes["history"][0]["changes"] = []
        cases["changes"] = empty_changes

        too_much_history = release_document()
        first = too_much_history["history"][0]
        too_much_history["history"] = [copy.deepcopy(first) for _ in range(33)]
        cases["history"] = too_much_history

        for name, document in cases.items():
            with self.subTest(name=name):
                self.assert_invalid(document)


class RecipeReleaseBuildTests(unittest.TestCase):
    def test_source_bundle_rejects_files_above_hydration_limit(self) -> None:
        with isolated_root() as root:
            context = root / "adapters/demo"
            context.mkdir(parents=True)
            (context / "oversized.bin").write_bytes(b"four")
            previous = catalog_index.MAX_SOURCE_FILE_BYTES
            catalog_index.MAX_SOURCE_FILE_BYTES = 3
            try:
                with self.assertRaisesRegex(
                    SystemExit, "source bundle file exceeds the Git blob hydration limit"
                ):
                    catalog_index.source_bundle(context)
            finally:
                catalog_index.MAX_SOURCE_FILE_BYTES = previous


if __name__ == "__main__":
    unittest.main()
