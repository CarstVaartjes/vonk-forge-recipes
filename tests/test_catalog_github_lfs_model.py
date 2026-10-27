from __future__ import annotations

import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from vonk_forge_contracts import ModelDefinition, document_sha256

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/catalog-github-lfs-model"
REVISION = "45c03facbf848e0c743091b7d9c17952096ff908"
REPOSITORY = "https://github.com/NVlabs/GR00T-WholeBodyControl"
OID = "84afc7c229473351a24b0a7d79fc47be9dbb81bd12774285f2f60a3c0e9028df"
SIZE = 7761041
PATH = "motionbricks/out/G1-clip.ckpt"


def pointer(oid: str = OID, size: int = SIZE) -> bytes:
    return (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"
    ).encode()


FakeResponse = io.BytesIO  # a context manager with read(), like urlopen


class FakeGitHub:
    """Recorded-shape answers; every request is kept so tests can inspect it."""

    def __init__(self, files: dict[str, bytes], batch: dict | None = None) -> None:
        self.files = files
        self.batch = batch
        self.requests: list = []

    def __call__(self, request, timeout: float | None = None):
        self.requests.append(request)
        url = request.full_url
        if url.startswith("https://raw.githubusercontent.com/"):
            prefix = f"https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/{REVISION}/"
            body = self.files.get(url.removeprefix(prefix))
            if not url.startswith(prefix) or body is None:
                raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
            return FakeResponse(body)
        if url == f"{REPOSITORY}.git/info/lfs/objects/batch":
            assert self.batch is not None
            return FakeResponse(json.dumps(self.batch).encode())
        raise AssertionError(f"unexpected request to {url}")


class CatalogGithubLfsModelTests(unittest.TestCase):
    def _run(self, github: FakeGitHub, extra: tuple[str, ...] = (), paths=(PATH,)):
        namespace = runpy.run_path(str(TOOL))
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                str(TOOL),
                "--root", directory,
                "--repository", REPOSITORY,
                "--revision", REVISION,
                "--publisher", "nvidia",
                "--model-group-slug", "motionbricks",
                "--model-group-title", "MotionBricks",
                "--model-group-description", "Motion generation.",
                "--model-slug", "motionbricks",
                "--model-title", "MotionBricks",
                "--model-description", "Motion generation.",
                "--version-slug", "motionbricks-g1",
                "--version", "G1",
                "--precision", "fp32",
                "--quantization", "none",
                "--license-spdx", "Apache-2.0",
                "--license-url", "https://example.com/license",
                "--attribution", "NVIDIA",
                *[arg for path in paths for arg in ("--path", path, "--role", "weights")],
                *extra,
            ]  # fmt: skip
            with (
                patch.object(sys, "argv", argv),
                patch.dict(namespace["read_pointer"].__globals__, {"urlopen": github}),
            ):
                try:
                    namespace["main"]()
                finally:
                    written = Path(directory) / "models/motionbricks-g1.json"
                    document = (
                        json.loads(written.read_text()) if written.exists() else None
                    )
        return document

    def test_pointer_becomes_an_anonymous_lfs_model(self) -> None:
        github = FakeGitHub({PATH: pointer()})
        document = self._run(github)
        assert document is not None
        ModelDefinition.model_validate(document)
        self.assertEqual(
            document["source"],
            {"provider": "github-lfs", "repository": REPOSITORY, "revision": REVISION},
        )
        self.assertFalse(document["requires_token"])
        (item,) = document["files"]
        self.assertEqual(
            (item["path"], item["sha256"], item["size_bytes"], item["roles"]),
            (PATH, OID, SIZE, ["weights"]),
        )
        self.assertEqual(len(github.requests), 1)
        for request in github.requests:
            names = {name.lower() for name in request.headers}
            self.assertFalse(names & {"authorization", "cookie"})
        self.assertEqual(len(document_sha256(document)), 64)

    def test_a_plain_git_file_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "vendor it in the recipe package"):
            self._run(FakeGitHub({PATH: b'{"plain": true}\n'}))
        with self.assertRaisesRegex(SystemExit, "vendor it in the recipe package"):
            self._run(FakeGitHub({PATH: b"x" * 5000}))

    def test_malformed_pointers_are_refused(self) -> None:
        for body in (
            pointer(oid="A" * 64),
            pointer(oid="a" * 63),
            pointer(size=0),
            pointer().replace(b"v1", b"v2"),
            pointer() + b"extra 1\n",
            pointer().rstrip(b"\n"),
        ):
            with self.subTest(body=body), self.assertRaises(SystemExit):
                self._run(FakeGitHub({PATH: body}))

    def test_a_missing_path_is_refused_with_the_status(self) -> None:
        with self.assertRaisesRegex(SystemExit, "HTTP 404"):
            self._run(FakeGitHub({}))

    def test_probe_batch_confirms_the_object_without_downloading(self) -> None:
        batch = {"objects": [{"oid": OID, "size": SIZE, "actions": {"download": {}}}]}
        github = FakeGitHub({PATH: pointer()}, batch)
        self._run(github, ("--probe-batch",))
        probe = github.requests[-1]
        self.assertEqual(probe.get_method(), "POST")
        self.assertEqual(
            json.loads(probe.data),
            {
                "operation": "download",
                "transfers": ["basic"],
                "objects": [{"oid": OID, "size": SIZE}],
            },
        )

    def test_probe_batch_refuses_a_missing_object(self) -> None:
        batch = {"objects": [{"oid": OID, "size": SIZE, "error": {"code": 404}}]}
        with self.assertRaisesRegex(SystemExit, "not available"):
            self._run(FakeGitHub({PATH: pointer()}, batch), ("--probe-batch",))

    def test_paths_and_roles_must_pair(self) -> None:
        github = FakeGitHub({PATH: pointer()})
        with self.assertRaisesRegex(SystemExit, "one --role for every --path"):
            self._run(github, ("--path", "other"))
        self.assertEqual(github.requests, [])


if __name__ == "__main__":
    unittest.main()
