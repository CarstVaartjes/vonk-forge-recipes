from __future__ import annotations

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/catalog-hf-model"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CatalogHfModelSafetyTests(unittest.TestCase):
    def _arguments(self, root: Path) -> list[str]:
        return [
            str(TOOL),
            "--root",
            str(root),
            "--repository",
            "owner/model",
            "--revision",
            "a" * 40,
            "--publisher",
            "owner",
            "--model-group-slug",
            "test-group",
            "--model-group-title",
            "Test group",
            "--model-group-description",
            "Test group.",
            "--model-slug",
            "test-model",
            "--model-title",
            "Test model",
            "--model-description",
            "Test model.",
            "--version-slug",
            "test-version",
            "--version",
            "Test version",
            "--precision",
            "bf16",
            "--quantization",
            "none",
            "--license-spdx",
            "MIT",
            "--license-url",
            "https://example.com/license",
            "--attribution",
            "Example",
            "--capability",
            "text-generation",
        ]

    def test_tree_follows_every_pagination_link(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        pages = {
            "https://example.test/page-1": (
                [{"path": "first"}],
                "https://example.test/page-2",
            ),
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
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "argv", self._arguments(Path(directory))),
            patch.dict(
                namespace["main"].__globals__,
                {
                    "_get_json": lambda _url: {"sha": "a" * 40},
                    "_get_tree": lambda _url: entries,
                },
            ),
        ):
            self.assertEqual(namespace["main"](), 0)
            model = load(Path(directory) / "models/test-version.json")
            self.assertEqual(len(model["files"]), 419)
            self.assertFalse(model["requires_token"])
            self.assertEqual(model["capabilities"], ["text-generation"])

    def test_artifact_ids_include_full_path_for_repeated_identical_files(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [
            {
                "type": "file",
                "path": f"traces/run-{index}/result.json",
                "size": 2,
                "lfs": {"oid": "b" * 64},
            }
            for index in range(3)
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "argv", self._arguments(Path(directory))),
            patch.dict(
                namespace["main"].__globals__,
                {
                    "_get_json": lambda _url: {"sha": "a" * 40},
                    "_get_tree": lambda _url: entries,
                },
            ),
        ):
            self.assertEqual(namespace["main"](), 0)
            model = load(Path(directory) / "models/test-version.json")
            ids = [item["id"] for item in model["files"]]
            self.assertEqual(len(ids), len(set(ids)))

    def _split_tree(self, count: int = 3, base: str = "model-00005.safetensors"):
        return [
            {
                "type": "file",
                "path": f"{base}.part{index:02d}",
                "size": 10 + index,
                "lfs": {"oid": f"{index + 1:064x}"},
            }
            for index in range(count)
        ] + [{"type": "file", "path": "x.bin", "size": 1, "lfs": {"oid": "f" * 64}}]

    def _run_main(self, entries, extra=(), manifest: bytes | None = None):
        namespace = runpy.run_path(str(TOOL))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "argv", [*self._arguments(Path(directory)), *extra]),
            patch.dict(
                namespace["main"].__globals__,
                {
                    "_get_json": lambda _url: {"sha": "a" * 40},
                    "_get_tree": lambda _url: entries,
                    "_get_bytes": lambda _url: manifest or self.fail("no download"),
                },
            ),
        ):
            namespace["main"]()
            return load(Path(directory) / "models/test-version.json")

    def test_split_parts_become_one_file_with_the_assembled_digest(self) -> None:
        whole = "9" * 64
        model = self._run_main(
            self._split_tree(),
            ["--assembled-sha256", f"model-00005.safetensors={whole}"],
        )
        by_path = {item["path"]: item for item in model["files"]}
        self.assertEqual(sorted(by_path), ["model-00005.safetensors", "x.bin"])
        entry = by_path["model-00005.safetensors"]
        self.assertEqual(entry["sha256"], whole)
        self.assertEqual(entry["size_bytes"], 10 + 11 + 12)
        self.assertEqual(entry["roles"], ["weights"])
        self.assertEqual(
            entry["parts"],
            [
                {
                    "path": f"model-00005.safetensors.part{index:02d}",
                    "sha256": f"{index + 1:064x}",
                    "size_bytes": 10 + index,
                }
                for index in range(3)
            ],
        )
        self.assertNotIn("parts", by_path["x.bin"])

    def test_split_digest_comes_from_the_repository_manifest(self) -> None:
        whole = "8" * 64
        manifest = f"{whole}  ./model-00005.safetensors\n".encode()
        entries = self._split_tree() + [
            {"type": "file", "path": "sha256-manifest.txt", "size": len(manifest)}
        ]
        namespace = runpy.run_path(str(TOOL))
        with patch.dict(
            namespace["inventory"].__globals__, {"_get_bytes": lambda _u: manifest}
        ):
            artifacts = namespace["inventory"]("o/n", "a" * 40, entries)
        entry = next(a for a in artifacts if a["path"] == "model-00005.safetensors")
        self.assertEqual(entry["sha256"], whole)

    def test_split_without_an_assembled_digest_is_refused(self) -> None:
        with self.assertRaisesRegex(
            SystemExit, "--assembled-sha256.*refusing to guess"
        ):
            self._run_main(self._split_tree())

    def test_split_refuses_gaps_conflicts_and_unrelated_digests(self) -> None:
        tree = self._split_tree(3)
        del tree[1]
        with self.assertRaisesRegex(SystemExit, "missing or repeated parts"):
            self._run_main(
                tree, ["--assembled-sha256", f"model-00005.safetensors={'9' * 64}"]
            )
        both = self._split_tree() + [
            {
                "type": "file",
                "path": "model-00005.safetensors",
                "size": 1,
                "lfs": {"oid": "e" * 64},
            }
        ]
        with self.assertRaisesRegex(SystemExit, "both whole and as parts"):
            self._run_main(both)
        with self.assertRaisesRegex(SystemExit, "not split sets"):
            self._run_main(
                self._split_tree(),
                [
                    "--assembled-sha256",
                    f"model-00005.safetensors={'9' * 64}",
                    "--assembled-sha256",
                    f"x.bin={'9' * 64}",
                ],
            )
        with self.assertRaisesRegex(SystemExit, "expects PATH"):
            self._run_main(
                self._split_tree(), ["--assembled-sha256", "model-00005.safetensors=zz"]
            )

    def test_a_lone_part_name_is_an_ordinary_file(self) -> None:
        model = self._run_main(self._split_tree(1))
        self.assertIn(
            "model-00005.safetensors.part00", [f["path"] for f in model["files"]]
        )

    def test_refuses_large_non_lfs_files_before_downloading(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": "large.bin", "size": 16 * 1024 * 1024 + 1}]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "argv", self._arguments(Path(directory))),
            patch.dict(
                namespace["main"].__globals__,
                {
                    "_get_json": lambda _url: {"sha": "a" * 40},
                    "_get_tree": lambda _url: entries,
                    "_get_bytes": lambda _url: self.fail("must not download"),
                },
            ),
            self.assertRaisesRegex(SystemExit, "refusing to download"),
        ):
            namespace["main"]()

    def test_gated_non_lfs_file_reports_actionable_authentication_error(self) -> None:
        namespace = runpy.run_path(str(TOOL))
        entries = [{"type": "file", "path": "config.json", "size": 2}]

        def denied(_url: str) -> bytes:
            error = HTTPError(_url, 401, "Unauthorized", {}, None)
            error.close()
            raise error

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "argv", self._arguments(Path(directory))),
            patch.dict(
                namespace["main"].__globals__,
                {
                    "_get_json": lambda _url: {"sha": "a" * 40},
                    "_get_tree": lambda _url: entries,
                    "_get_bytes": denied,
                },
            ),
            self.assertRaisesRegex(
                SystemExit, r"config\.json \(HTTP 401\); authenticate to Hugging Face"
            ),
        ):
            namespace["main"]()

    def test_catalog_command_exposes_model_authority_options(self) -> None:
        result = __import__("subprocess").run(
            [sys.executable, str(TOOL), "--help"], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--publisher",
            "--version-slug",
            "--requires-token",
            "--quantization",
        ):
            self.assertIn(option, result.stdout)

    def test_current_qwen_snapshot_has_unique_complete_files(self) -> None:
        version = load(ROOT / "models/qwen3-8-27b-fp8-017b9c7a.json")
        files = version["files"]
        self.assertGreater(len(files), 70)
        self.assertEqual(len({item["id"] for item in files}), len(files))
        self.assertEqual(len({item["path"] for item in files}), len(files))

    def test_qwen_dense_models_advertise_native_multimodal_capabilities(self) -> None:
        for slug in (
            "qwen3-5-9b-c2022362",
            "qwen3-6-27b-6a9e13bd",
            "qwen3-8-27b-1d4bf0f2",
        ):
            model = load(ROOT / f"models/{slug}.json")
            capabilities = set(model["capabilities"])
            self.assertTrue(
                capabilities & {"text-generation", "image-understanding"}, slug
            )


if __name__ == "__main__":
    unittest.main()
