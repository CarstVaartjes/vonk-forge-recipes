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

    def test_tree_follows_every_pagination_link(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        pages = {
            "https://example.test/page-1": ([{"path": "first"}], "https://example.test/page-2"),
            "https://example.test/page-2": ([{"path": "second"}], None),
        }
        with patch.dict(
            namespace["_get_tree"].__globals__,
            {"_get_json_page": lambda url: pages[url]},
        ):
            self.assertEqual(
                namespace["_get_tree"]("https://example.test/page-1"),
                [{"path": "first"}, {"path": "second"}],
            )

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

    def test_artifact_ids_include_full_path_for_repeated_identical_files(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        digest = "b" * 64
        entries = [
            {
                "type": "file",
                "path": f"traces/run-{index}/result.json",
                "size": 2,
                "lfs": {"oid": digest},
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", self._arguments(Path(directory))
        ), patch.dict(
            namespace["main"].__globals__,
            {
                "_get_json": lambda _url: {"sha": "a" * 40},
                "_get_tree": lambda _url: entries,
            },
        ):
            self.assertEqual(namespace["main"](), 0)
            version = json.loads(
                (Path(directory) / "model-versions/test-version.json").read_text()
            )
            ids = [artifact["id"] for artifact in version["artifacts"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_refuses_large_non_lfs_files_before_downloading(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": "large.bin", "size": 16 * 1024 * 1024 + 1}]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", self._arguments(Path(directory))
        ), patch.dict(namespace["main"].__globals__, {"_get_json": lambda _url: {"sha": "a" * 40}, "_get_tree": lambda _url: entries, "_get_bytes": lambda _url: self.fail("must not download")}):
            with self.assertRaisesRegex(SystemExit, "refusing to download"):
                namespace["main"]()

    def test_gated_non_lfs_file_fails_with_actionable_authentication_error(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": "config.json", "size": 2}]

        def denied(_url: str) -> bytes:
            error = HTTPError(_url, 401, "Unauthorized", {}, None)
            error.close()
            raise error

        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", self._arguments(Path(directory))
        ), patch.dict(
            namespace["main"].__globals__,
            {
                "_get_json": lambda _url: {"sha": "a" * 40},
                "_get_tree": lambda _url: entries,
                "_get_bytes": denied,
            },
        ):
            with self.assertRaisesRegex(
                SystemExit,
                r"config\.json \(HTTP 401\); authenticate to Hugging Face",
            ):
                namespace["main"]()

    def test_qwen38_fp8_catalog_closes_the_complete_snapshot(self) -> None:
        version = json.loads(
            (ROOT / "model-versions/qwen3-8-27b-fp8-017b9c7a.json").read_text()
        )
        recipe = json.loads(
            (ROOT / "recipes/qwen3-8-27b-fp8-vllm-single.json").read_text()
        )
        artifacts = version["artifacts"]
        expected_bytes = 30_890_049_597

        self.assertEqual(len(artifacts), 80)
        self.assertEqual(sum(item["download_bytes"] for item in artifacts), expected_bytes)
        self.assertEqual(version["sizes"]["download_bytes"], expected_bytes)
        self.assertEqual(recipe["artifacts"][0]["download_bytes"], expected_bytes)
        self.assertIn("video_preprocessor_config.json", {item["path"] for item in artifacts})

    def test_qwen_dense_models_advertise_native_multimodal_capabilities(self) -> None:
        for slug in ("qwen3-5-9b", "qwen3-6-27b", "qwen3-8-27b"):
            with self.subTest(slug=slug):
                model = json.loads((ROOT / f"models/{slug}.json").read_text())
                tags = set(model["metadata"]["tags"])
                self.assertTrue({"multimodal", "vision", "reasoning", "agentic"} <= tags)
        qwen35 = json.loads((ROOT / "models/qwen3-5-9b.json").read_text())
        self.assertNotIn("moe", qwen35["metadata"]["tags"])
        self.assertIn("dense", qwen35["architecture"].lower())


if __name__ == "__main__":
    unittest.main()
