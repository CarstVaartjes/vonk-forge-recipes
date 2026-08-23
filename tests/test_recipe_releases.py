from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/build-catalog-index"
LOADER = importlib.machinery.SourceFileLoader("build_catalog_index", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
catalog_index = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = catalog_index
LOADER.exec_module(catalog_index)

CURRENT_DIGEST = "a" * 64
OLDER_DIGEST = "b" * 64


def release_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "recipe": {"publisher": "example", "slug": "demo"},
        "version": "2.0.0",
        "released_at": "2026-08-23",
        "history": [
            {
                "version": "2.0.0",
                "released_at": "2026-08-23",
                "recipe_content_sha256": CURRENT_DIGEST,
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
                "recipe_content_sha256": OLDER_DIGEST,
                "upgrade_effect": "reinstall",
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


def write_release(root: Path, document: dict[str, object]) -> Path:
    directory = root / "recipe-releases"
    directory.mkdir(exist_ok=True)
    path = directory / "demo.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class RecipeReleaseValidationTests(unittest.TestCase):
    def validate(self, document: dict[str, object]) -> dict[str, object]:
        with isolated_root() as root:
            path = write_release(root, document)
            return catalog_index.recipe_release(
                path,
                publisher="example",
                slug="demo",
                recipe_digest=CURRENT_DIGEST,
            )

    def assert_invalid(self, document: dict[str, object], detail: str) -> None:
        with self.assertRaisesRegex(SystemExit, detail):
            self.validate(document)

    def test_accepts_current_digest_and_newest_first_history(self) -> None:
        release = self.validate(release_document())

        self.assertEqual(release["version"], "2.0.0")
        self.assertEqual(release["released_at"], "2026-08-23")
        history = release["history"]
        assert isinstance(history, list)
        self.assertEqual(
            [item["recipe_content_sha256"] for item in history],
            [CURRENT_DIGEST, OLDER_DIGEST],
        )

    def test_rejects_current_digest_mismatch(self) -> None:
        document = release_document()
        document["history"][0]["recipe_content_sha256"] = "c" * 64

        self.assert_invalid(document, "current recipe digest")

    def test_rejects_identity_mismatch(self) -> None:
        document = release_document()
        document["recipe"]["slug"] = "different"

        self.assert_invalid(document, "identity does not match")

    def test_rejects_unsorted_or_duplicate_history(self) -> None:
        cases: dict[str, tuple[dict[str, object], str]] = {}

        unsorted = release_document()
        unsorted["history"][1]["version"] = "3.0.0"
        cases["unsorted version"] = (unsorted, "semantic version")

        duplicate_version = release_document()
        duplicate_version["history"][1]["version"] = "2.0.0"
        cases["duplicate version"] = (duplicate_version, "unique semantic version")

        duplicate_digest = release_document()
        duplicate_digest["history"][1]["recipe_content_sha256"] = CURRENT_DIGEST
        cases["duplicate digest"] = (duplicate_digest, "unique digest")

        unsorted_date = release_document()
        unsorted_date["history"][1]["released_at"] = "2026-08-24"
        cases["unsorted date"] = (unsorted_date, "newest-first by released_at")

        for name, (document, detail) in cases.items():
            with self.subTest(name=name):
                self.assert_invalid(document, detail)

    def test_rejects_invalid_date_category_and_upgrade_effect(self) -> None:
        cases: dict[str, tuple[dict[str, object], str]] = {}

        invalid_date = release_document()
        invalid_date["released_at"] = "2026-02-30"
        cases["date"] = (invalid_date, "ISO 8601 calendar date")

        invalid_category = release_document()
        invalid_category["history"][0]["changes"][0]["kind"] = "marketing"
        cases["category"] = (invalid_category, "kind is unsupported")

        invalid_effect = release_document()
        invalid_effect["history"][0]["upgrade_effect"] = "redeploy"
        cases["upgrade effect"] = (invalid_effect, "upgrade_effect is unsupported")

        for name, (document, detail) in cases.items():
            with self.subTest(name=name):
                self.assert_invalid(document, detail)

    def test_rejects_bounded_text_and_collection_overflows(self) -> None:
        cases: dict[str, tuple[dict[str, object], str]] = {}

        long_summary = release_document()
        long_summary["history"][0]["changes"][0]["summary"] = "s" * 161
        cases["summary"] = (long_summary, "1..160 characters")

        long_details = release_document()
        long_details["history"][0]["changes"][0]["details"] = "d" * 1001
        cases["details"] = (long_details, "1..1000 characters")

        too_many_references = release_document()
        too_many_references["history"][0]["changes"][0]["references"] = [
            f"https://example.com/{index}" for index in range(9)
        ]
        cases["references"] = (too_many_references, "1..8 unique URLs")

        empty_changes = release_document()
        empty_changes["history"][0]["changes"] = []
        cases["changes"] = (empty_changes, "changes must contain 1..16 entries")

        too_much_history = release_document()
        first = too_much_history["history"][0]
        too_much_history["history"] = [copy.deepcopy(first) for _ in range(33)]
        cases["history"] = (too_much_history, "history must contain 1..32 releases")

        for name, (document, detail) in cases.items():
            with self.subTest(name=name):
                self.assert_invalid(document, detail)


class RecipeReleaseBuildTests(unittest.TestCase):
    def recipe_fixture(self, root: Path) -> dict[str, object]:
        context = root / "adapters/demo"
        context.mkdir(parents=True)
        (context / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        archive, _, digest = catalog_index.source_bundle(context)
        return {
            "identity": {"publisher": "example", "slug": "demo"},
            "build": {
                "context": {
                    "path": "adapters/demo",
                    "sha256": digest,
                    "expected_bytes": len(archive),
                }
            },
        }

    def test_build_rejects_missing_sidecar(self) -> None:
        with isolated_root() as root:
            recipes = root / "recipes"
            recipes.mkdir()
            recipe = self.recipe_fixture(root)
            (recipes / "demo.json").write_text(json.dumps(recipe), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "recipe release is missing"):
                catalog_index.build()

    def test_build_rejects_orphan_sidecar(self) -> None:
        with isolated_root() as root:
            recipes = root / "recipes"
            recipes.mkdir()
            recipe = self.recipe_fixture(root)
            (recipes / "demo.json").write_text(json.dumps(recipe), encoding="utf-8")
            digest = hashlib.sha256(catalog_index.canonical(recipe)).hexdigest()
            release = release_document()
            release["history"][0]["recipe_content_sha256"] = digest
            write_release(root, release)
            (root / "recipe-releases/orphan.json").write_text(
                json.dumps(release), encoding="utf-8"
            )

            with self.assertRaisesRegex(SystemExit, "orphan recipe release"):
                catalog_index.build()


if __name__ == "__main__":
    unittest.main()
