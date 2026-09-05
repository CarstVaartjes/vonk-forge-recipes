from __future__ import annotations

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/catalog-hf-model"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CatalogHfModelSafetyTests(unittest.TestCase):
    def _arguments(self, root: Path) -> list[str]:
        return [
            str(TOOL), "--root", str(root), "--repository", "owner/model",
            "--revision", "a" * 40, "--publisher", "owner",
            "--model-group-slug", "test-group", "--model-group-title", "Test group",
            "--model-group-description", "Test group.", "--model-slug", "test-model",
            "--model-title", "Test model", "--model-description", "Test model.",
            "--architecture", "test-architecture", "--version-slug", "test-version",
            "--version", "Test version", "--precision", "bf16", "--quantization", "none",
            "--parameters-total", "1", "--parameters-active", "1", "--context-tokens", "1",
            "--license-spdx", "MIT", "--license-url", "https://example.com/license",
            "--attribution", "Example",
        ]

    def test_tree_follows_every_pagination_link(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        pages = {
            "https://example.test/page-1": ([{"path": "first"}], "https://example.test/page-2"),
            "https://example.test/page-2": ([{"path": "second"}], None),
        }
        with patch.dict(namespace["_get_tree"].__globals__, {"_get_json_page": lambda url: pages[url]}):
            self.assertEqual(namespace["_get_tree"]("https://example.test/page-1"), [{"path": "first"}, {"path": "second"}])

    def test_preserves_large_complete_file_inventories(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": f"model-{index:05d}.safetensors", "size": index + 1, "lfs": {"oid": f"{index + 1:064x}"}} for index in range(419)]
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "argv", self._arguments(Path(directory))), patch.dict(namespace["main"].__globals__, {"_get_json": lambda _url: {"sha": "a" * 40}, "_get_tree": lambda _url: entries}):
            self.assertEqual(namespace["main"](), 0)
            model = load(Path(directory) / "models/test-version.json")
            self.assertEqual(len(model["files"]), 419)

    def test_artifact_ids_include_full_path_for_repeated_identical_files(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": f"traces/run-{index}/result.json", "size": 2, "lfs": {"oid": "b" * 64}} for index in range(3)]
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "argv", self._arguments(Path(directory))), patch.dict(namespace["main"].__globals__, {"_get_json": lambda _url: {"sha": "a" * 40}, "_get_tree": lambda _url: entries}):
            self.assertEqual(namespace["main"](), 0)
            model = load(Path(directory) / "models/test-version.json")
            ids = [item["id"] for item in model["files"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_refuses_large_non_lfs_files_before_downloading(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": "large.bin", "size": 16 * 1024 * 1024 + 1}]
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "argv", self._arguments(Path(directory))), patch.dict(namespace["main"].__globals__, {"_get_json": lambda _url: {"sha": "a" * 40}, "_get_tree": lambda _url: entries, "_get_bytes": lambda _url: self.fail("must not download")}):
            with self.assertRaisesRegex(SystemExit, "refusing to download"):
                namespace["main"]()

    def test_gated_non_lfs_file_reports_actionable_authentication_error(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": "config.json", "size": 2}]
        def denied(_url: str) -> bytes:
            error = HTTPError(_url, 401, "Unauthorized", {}, None)
            error.close()
            raise error
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "argv", self._arguments(Path(directory))), patch.dict(namespace["main"].__globals__, {"_get_json": lambda _url: {"sha": "a" * 40}, "_get_tree": lambda _url: entries, "_get_bytes": denied}):
            with self.assertRaisesRegex(SystemExit, r"config\.json \(HTTP 401\); authenticate to Hugging Face"):
                namespace["main"]()

    def test_catalog_command_exposes_model_authority_options(self) -> None:
        result = __import__("subprocess").run([sys.executable, str(TOOL), "--help"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--publisher", "--version-slug", "--architecture", "--quantization"):
            self.assertIn(option, result.stdout)

    def test_current_qwen_snapshot_has_unique_complete_files(self) -> None:
        version = load(ROOT / "models/qwen3-8-27b-fp8-017b9c7a.json")
        files = version["files"]
        self.assertGreater(len(files), 70)
        self.assertEqual(len({item["id"] for item in files}), len(files))
        self.assertEqual(len({item["path"] for item in files}), len(files))

    def test_qwen_dense_models_advertise_native_multimodal_capabilities(self) -> None:
        for slug in ("qwen3-5-9b-c2022362", "qwen3-6-27b-6a9e13bd", "qwen3-8-27b-1d4bf0f2"):
            model = load(ROOT / f"models/{slug}.json")
            capabilities = {fact["capability"] for fact in model["capabilities"]["facts"] if fact["support"] == "supported"}
            self.assertTrue(capabilities & {"text-generation", "image-understanding"}, slug)


if __name__ == "__main__":
    unittest.main()
