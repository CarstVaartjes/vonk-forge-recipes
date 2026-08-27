from __future__ import annotations

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/catalog-hf-model"


class CatalogHfModelSafetyTests(unittest.TestCase):
    def _arguments(self, root: Path) -> list[str]:
        return [
            str(TOOL), "--root", str(root), "--repository", "owner/model",
            "--revision", "a" * 40, "--publisher", "owner",
            "--model-group-slug", "test-group", "--model-group-title", "Test group",
            "--model-group-description", "Test group.", "--model-slug", "test-model",
            "--model-title", "Test model", "--model-description", "Test model.",
            "--architecture", "TestArchitecture", "--version-slug", "test-version",
            "--version", "Test version", "--precision", "bf16", "--quantization", "none",
            "--parameters-total", "1", "--parameters-active", "1", "--context-tokens", "1",
            "--license-spdx", "MIT", "--license-url", "https://example.com/license",
            "--attribution", "Example",
        ]

    def test_preserves_large_complete_file_inventories(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [
            {
                "type": "file",
                "path": f"model-{index:05d}.safetensors",
                "size": index + 1,
                "lfs": {"oid": f"{index + 1:064x}"},
            }
            for index in range(419)
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", self._arguments(Path(directory))
        ), patch.dict(namespace["main"].__globals__, {"_get_json": lambda _url: {"sha": "a" * 40}, "_get_tree": lambda _url: entries}):
            self.assertEqual(namespace["main"](), 0)
            version = json.loads(
                (Path(directory) / "model-versions/test-version.json").read_text()
            )
            self.assertEqual(len(version["artifacts"]), 419)

    def test_refuses_large_non_lfs_files_before_downloading(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": "large.bin", "size": 16 * 1024 * 1024 + 1}]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", self._arguments(Path(directory))
        ), patch.dict(namespace["main"].__globals__, {"_get_json": lambda _url: {"sha": "a" * 40}, "_get_tree": lambda _url: entries, "_get_bytes": lambda _url: self.fail("must not download")}):
            with self.assertRaisesRegex(SystemExit, "refusing to download"):
                namespace["main"]()


if __name__ == "__main__":
    unittest.main()
