"""A kit that provides a prebuilt image is run as published (tools/kit-pins images rule)."""

from __future__ import annotations

import http.client
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from test_kit_pins import KIT
from upstream_fakes import patched_open

from recipe_tooling.kit_pins import archive as kit_archive
from recipe_tooling.kit_pins import check as kit_check
from recipe_tooling.kit_pins import image_scan as kit_image_scan
from recipe_tooling.kit_pins import model as kit_model
from recipe_tooling.kit_pins import sources as kit_sources
from recipe_tooling.kit_pins import upstream as kit_upstream

COMMIT = "a" * 40
IMAGE = "ghcr.io/creator/kit-image"
DIGEST = "sha256:" + "b" * 64


def archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"kit-{COMMIT[:7]}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def scan(files: dict[str, str], scope: str = "") -> list[str]:
    return [f["reference"] for f in kit_image_scan.scan_archive(archive(files), scope)]


class ScanTests(unittest.TestCase):
    def test_finds_the_images_a_kit_runs_or_pulls(self) -> None:
        found = scan(
            {
                "Dockerfile": f"FROM {IMAGE}@{DIGEST}\n",
                "scripts/config.sh": 'IMG="${IMG:-ghcr.io/creator/other:v1}"\n',
                "start.sh": "docker run --rm --gpus all -v /a/b:/c creator/hub-image:tag python\n",
                "compose.yaml": "services:\n  x:\n    image: quay.io/creator/third:1\n",
                "README.md": "docker pull ghcr.io/creator/readme:latest\nsee ghcr.io/creator/prose:1\n",
            }
        )
        assert sorted(found) == sorted(
            [
                f"{IMAGE}@{DIGEST}",
                "ghcr.io/creator/other:v1",
                "creator/hub-image:tag",
                "quay.io/creator/third:1",
                "ghcr.io/creator/readme:latest",
            ]
        )

    def test_engines_bases_and_documentation_are_not_creator_images(self) -> None:
        found = scan(
            {
                "Dockerfile": "FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n",
                "a/Dockerfile": "FROM nvcr.io/nvidia/pytorch:26.07-py3\n",
                "b/Dockerfile": "FROM lmsysorg/sglang:v1\n",
                "docs/run.sh": f"docker run {IMAGE}:x\n",
                "tests/run.sh": f"docker run {IMAGE}:x\n",
                "run.sh": "# docker run ghcr.io/creator/commented:x\n",
                "paths.sh": "docker run -v models/foo:/data -e HF=org/model:rev img\n",
                "model.sh": "MODEL_IMAGE_ID=Qwen/Qwen-Image:file\n",
            }
        )
        assert found == [], found

    def test_a_monorepo_kit_is_scoped_to_the_cited_directory(self) -> None:
        files = {
            "one/start.sh": f"docker run {IMAGE}:a\n",
            "two/start.sh": "docker run ghcr.io/creator/two:a\n",
        }
        assert scan(files, "two/") == ["ghcr.io/creator/two:a"]
        recipe = {
            "provenance": {
                "source_reference": f"https://github.com/{KIT}/blob/{COMMIT}/two/README.md"
            }
        }
        assert kit_image_scan.kit_scope(recipe) == "two/"
        recipe["provenance"]["source_reference"] = (
            f"https://github.com/{KIT}/tree/{COMMIT}"
        )
        assert kit_image_scan.kit_scope(recipe) == ""


def recipe(root: Path, dockerfile: str, base: str | None = None) -> dict[str, Any]:
    adapter = root / "adapters/x"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "Dockerfile").write_text(dockerfile)
    build: dict[str, Any] = {"context": {"path": "adapters/x"}}
    if base:
        build["base_image"] = {"repository": base, "digest": "b" * 64}
    return {
        "identity": {"slug": "x"},
        "provenance": {"source_reference": f"https://github.com/{KIT}/tree/{COMMIT}"},
        "execution": {"build": build},
    }


class DownloadScanTests(unittest.TestCase):
    def _download(self, body: bytes) -> list[str]:
        response = io.BytesIO(body)
        with (
            patched_open(mock.Mock(return_value=response)),
            mock.patch.object(kit_archive, "MAX_ARCHIVE_BYTES", 1),
        ):
            return [f["reference"] for f in kit_image_scan.scan_download("o/r", COMMIT)]

    def test_an_archive_larger_than_the_vendoring_cap_is_scanned_as_it_streams(
        self,
    ) -> None:
        body = archive({"Dockerfile": f"FROM {IMAGE}@{DIGEST}\n"})
        assert self._download(body) == [f"{IMAGE}@{DIGEST}"]

    def test_a_truncated_download_fails_with_the_url(self) -> None:
        body = archive({"Dockerfile": f"FROM {IMAGE}@{DIGEST}\n"})
        with pytest.raises(kit_model.KitError, match="cannot scan .*tar.gz"):
            self._download(body[: len(body) // 2])


class _Dropping(io.BytesIO):
    """A response that returns some bytes, then fails like a broken connection."""

    def __init__(self, body: bytes, error: BaseException) -> None:
        super().__init__(body)
        self._error = error

    def read(self, size: int | None = -1) -> bytes:
        data = super().read(size)
        if not data or (size is not None and size > 0 and len(data) < size):
            raise self._error
        return data


MID_STREAM_ERRORS = [
    TimeoutError("timed out"),
    ConnectionResetError("reset by peer"),
    http.client.IncompleteRead(b"x", 10),
]


class MidStreamFailureTests(unittest.TestCase):
    def test_a_download_that_breaks_mid_stream_is_a_kit_error_with_the_url(
        self,
    ) -> None:
        body = archive({"Dockerfile": f"FROM {IMAGE}@{DIGEST}\n"})
        for error in MID_STREAM_ERRORS:
            with self.subTest(error=type(error).__name__):
                scan = _Dropping(body[: len(body) // 2], error)
                fetch = _Dropping(body[: len(body) // 2], error)
                with (
                    patched_open(mock.Mock(return_value=scan)),
                    pytest.raises(kit_model.KitError, match="cannot scan .*tar.gz"),
                ):
                    kit_image_scan.scan_download("o/r", COMMIT)
                with (
                    patched_open(
                        mock.Mock(
                            side_effect=[
                                _Dropping(fetch.getvalue(), error) for _ in range(3)
                            ]
                        )
                    ),
                    pytest.raises(kit_model.KitError, match="cannot fetch .*tar.gz"),
                ):
                    kit_archive.download_archive("o/r", COMMIT, lambda _seconds: None)

    def test_a_broken_download_is_retried_until_it_completes(self) -> None:
        body = archive({"Dockerfile": f"FROM {IMAGE}@{DIGEST}\n"})
        answers = [_Dropping(body[:5], TimeoutError("timed out")), io.BytesIO(body)]
        pauses: list[float] = []
        with patched_open(mock.Mock(side_effect=answers)):
            data = kit_archive.download_archive("o/r", COMMIT, pauses.append)
        assert (data, len(pauses)) == (body, 1)

    def test_scanning_does_not_retain_the_members_it_has_passed(self) -> None:
        files = {f"d/{n}.txt": "x" for n in range(200)}
        files["Dockerfile"] = f"FROM {IMAGE}@{DIGEST}\n"
        body = archive(files)
        retained = []
        real = tarfile.open

        def spy(*args: Any, **kwargs: Any) -> Any:
            handle = real(*args, **kwargs)
            retained.append(handle)
            return handle

        with mock.patch.object(tarfile, "open", spy):
            found = kit_image_scan.scan_stream(io.BytesIO(body))
        assert [f["reference"] for f in found] == [f"{IMAGE}@{DIGEST}"]
        assert len(retained[0].members) < 3


class CliSourceTests(unittest.TestCase):
    def test_the_cli_image_scan_streams_a_large_archive_once_per_commit_and_scope(
        self,
    ) -> None:
        body = archive({"sub/Dockerfile": f"FROM {IMAGE}@{DIGEST}\n"})
        opened: list[str] = []

        def urlopen(request: Any, timeout: float | None = None) -> io.BytesIO:
            opened.append(request.full_url)
            return io.BytesIO(body)

        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.dict(
                os.environ,
                {"VONK_UPSTREAM_CACHE": cache, "GITHUB_TOKEN": "unused"},
            ),
            patched_open(urlopen),
            mock.patch.object(kit_archive, "MAX_ARCHIVE_BYTES", 1),
        ):
            source = kit_sources.cli_source()
            first = kit_image_scan.kit_images(source, "o/r", COMMIT, "sub/")
            second = kit_image_scan.kit_images(source, "o/r", COMMIT, "sub/")
        assert [f["reference"] for f in first] == [f"{IMAGE}@{DIGEST}"]
        assert second == first
        assert opened == [f"https://codeload.github.com/o/r/tar.gz/{COMMIT}"]


class RuleTests(unittest.TestCase):
    def problems(
        self,
        dockerfile: str,
        referenced: list[str],
        exceptions: dict[str, str] | None = None,
        base: str | None = None,
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = recipe(root, dockerfile, base)
            found = [
                {
                    "repo": kit_image_scan.image_name(r.split(":")[0].split("@")[0]),
                    "reference": r,
                    "file": "start.sh",
                    "line": 1,
                }
                for r in referenced
            ]
            return kit_image_scan.image_problems(
                root, "x", document, found, exceptions or {}
            )

    def test_a_recipe_that_builds_from_the_kits_image_passes(self) -> None:
        assert self.problems(f"FROM {IMAGE}@{DIGEST}\n", [f"{IMAGE}:tag"]) == []

    def test_the_recorded_base_image_counts_as_running_it(self) -> None:
        assert self.problems("FROM scratch\n", [f"{IMAGE}:tag"], base=IMAGE) == []

    def test_a_self_build_beside_a_published_image_is_a_problem(self) -> None:
        problems = self.problems(
            "FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n", [f"{IMAGE}:tag"]
        )
        assert len(problems) == 1
        assert "kit provides prebuilt image" in problems[0]
        assert "policy/kit-image-exceptions.json" in problems[0]

    def test_a_reasoned_exception_allows_it(self) -> None:
        assert (
            self.problems(
                "FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n",
                [f"{IMAGE}:tag"],
                {"x": "the kit's image is another engine lane than this recipe"},
            )
            == []
        )

    def test_an_exception_needs_a_reason(self) -> None:
        assert (
            "no reason"
            in self.problems("FROM scratch\n", [f"{IMAGE}:tag"], {"x": " "})[0]
        )

    def test_a_kit_without_an_image_needs_nothing(self) -> None:
        assert (
            self.problems("FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n", []) == []
        )


class CheckTests(unittest.TestCase):
    def test_checks_every_recipe_once_per_kit_and_names_unknown_exceptions(
        self,
    ) -> None:
        calls: list[tuple[str, str, str]] = []

        def images(repo: str, commit: str, scope: str) -> list[dict[str, Any]]:
            calls.append((repo, commit, scope))
            return [
                {
                    "repo": IMAGE,
                    "reference": f"{IMAGE}:t",
                    "file": "start.sh",
                    "line": 3,
                }
            ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = recipe(root, f"FROM {IMAGE}@{DIGEST}\n")
            second = json.loads(json.dumps(first))
            second["identity"]["slug"] = "y"
            second["execution"]["build"]["context"]["path"] = "adapters/y"
            (root / "adapters/y").mkdir()
            (root / "adapters/y/Dockerfile").write_text("FROM scratch\n")
            root.joinpath("policy").mkdir(exist_ok=True)
            (root / kit_image_scan.EXCEPTIONS).write_text(
                json.dumps({"gone": "reason", "x": "was needed once"})
            )
            notices: list[str] = []
            found = kit_check.check_images(
                root,
                {"x": first, "y": second},
                kit_upstream.Upstream(
                    kit_upstream.GitHub(lambda path: {}), images=images
                ),
                notice=notices.append,
            )
        assert len(calls) == 1
        assert len(found) == 2, found
        assert notices == [
            "remove the policy/kit-image-exceptions.json entry of x: it runs the kit's image ghcr.io/creator/kit-image"
        ]
        assert any("'gone' is not a recipe" in p for p in found), found
        assert any(p.startswith("y: the kit provides") for p in found), found


if __name__ == "__main__":
    unittest.main()
