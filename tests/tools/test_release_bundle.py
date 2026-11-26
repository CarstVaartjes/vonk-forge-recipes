"""The release bundle builder and the asset synchronizer used by publication."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
from repo_root import ROOT


def load_tool(name: str) -> ModuleType:
    path = ROOT / "tools" / name
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


bundle_tool = load_tool("build-release-bundle")
sync_tool = load_tool("sync-release-assets")

FILES = {
    "catalog-index.json": b'{"kind":"recipe-library-index"}\n',
    "qualification-index.json": b"{}\n",
    "zeta-recipe.tar.gz": b"\x1f\x8bzeta",
    "alpha-recipe.tar.gz": b"\x1f\x8balpha",
    "coverage-2026-09-24.md": b"# coverage\n",
}


def write_release(directory: Path, files: dict[str, bytes] = FILES) -> Path:
    """A release directory shaped like tools/build-catalog-index --release-dir."""

    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (directory / name).write_bytes(payload)
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
            for name, payload in sorted(files.items())
        ),
        encoding="utf-8",
    )
    (directory / "SHA256SUMS.sigstore.json").write_text('{"bundle":1}')
    return directory


class ReleaseBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)

    def test_members_are_the_manifest_plus_signature_and_nothing_else(self) -> None:
        release = write_release(self.tmp / "release")
        output = self.tmp / "recipe-library.tar"
        bundle_tool.build(release, output)
        listed = {
            line.split("  ")[1]
            for line in (release / "SHA256SUMS").read_text().splitlines()
        }
        with tarfile.open(output) as archive:
            names = archive.getnames()
            assert names == sorted(names)
            assert set(names) == listed | {"SHA256SUMS", "SHA256SUMS.sigstore.json"}
            for member in archive.getmembers():
                assert member.isreg(), member.name
                assert "/" not in member.name
                assert (member.mode, member.uid, member.gid, member.mtime) == (
                    420,
                    0,
                    0,
                    0,
                )
                assert (member.uname, member.gname) == ("", "")
                extracted = archive.extractfile(member)
                assert extracted is not None
                assert extracted.read() == (release / member.name).read_bytes()

    def test_bytes_do_not_depend_on_directory_state(self) -> None:
        first = write_release(self.tmp / "first")
        second = write_release(self.tmp / "second", dict(reversed(FILES.items())))
        for path in second.iterdir():
            os.utime(path, (1_700_000_000, 1_700_000_000))
        one, two = self.tmp / "one.tar", self.tmp / "two.tar"
        bundle_tool.build(first, one)
        bundle_tool.build(second, two)
        assert one.read_bytes() == two.read_bytes()

    def test_names_beyond_ustar_are_refused_not_extended(self) -> None:
        exact = f"{'a' * 93}.tar.gz"
        release = write_release(self.tmp / "ok", {**FILES, exact: b"\x1f\x8bok"})
        bundle_tool.build(release, self.tmp / "ok.tar")
        with tarfile.open(self.tmp / "ok.tar") as archive:
            assert exact in archive.getnames()
            assert {member.type for member in archive.getmembers()} == {tarfile.REGTYPE}
        release = write_release(
            self.tmp / "long", {**FILES, f"{'a' * 94}.tar.gz": b"\x1f\x8blong"}
        )
        with pytest.raises(SystemExit):
            bundle_tool.build(release, self.tmp / "long.tar")

    def test_inconsistent_release_directories_are_refused(self) -> None:
        cases = {
            "unlisted file": lambda r: (r / "stray.txt").write_text("x"),
            "missing file": lambda r: (r / "catalog-index.json").unlink(),
            "changed file": lambda r: (r / "qualification-index.json").write_text("!"),
            "missing bundle signature": lambda r: (
                r / "SHA256SUMS.sigstore.json"
            ).unlink(),
            "symlink": lambda r: (
                (r / "alpha-recipe.tar.gz").unlink(),
                (r / "alpha-recipe.tar.gz").symlink_to(r / "catalog-index.json"),
            ),
        }
        for label, damage in cases.items():
            with self.subTest(label):
                release = write_release(self.tmp / label.replace(" ", "-"))
                damage(release)
                with pytest.raises(SystemExit):
                    bundle_tool.build(release, self.tmp / f"{label}.tar")

    def test_manifest_must_not_list_itself_or_repeat_names(self) -> None:
        digest = "0" * 64
        for text in (
            f"{digest}  SHA256SUMS\n",
            f"{digest}  a.json\n{digest}  a.json\n",
            f"{digest} a.json\n",
            f"{digest}  dir/a.json\n",
            "",
        ):
            with self.subTest(text=text), pytest.raises(SystemExit):
                bundle_tool.parse_checksums(text)


class FakeGitHub:
    """A release whose asset uploads can fail or land corrupted, behind ``gh``."""

    def __init__(self, failures: Sequence[str] = ()) -> None:
        self.assets: dict[str, dict[str, object]] = {}
        self.failures = list(failures)
        self.uploads: list[list[str]] = []

    def put(self, name: str, payload: bytes, digest: str | None = None) -> None:
        self.assets[name] = {
            "name": name,
            "size": len(payload),
            "state": "uploaded",
            "digest": digest or f"sha256:{hashlib.sha256(payload).hexdigest()}",
        }

    def __call__(self, arguments: Sequence[str]) -> str:
        arguments = list(arguments)
        if arguments[0] == "api":
            endpoint = arguments[1]
            if "/releases/tags/" in endpoint:
                return json.dumps({"id": 7})
            return json.dumps(
                list(self.assets.values()) if "page=1" in endpoint else []
            )
        if arguments[:2] == ["release", "upload"]:
            paths = [Path(item) for item in arguments if item.startswith("/")]
            self.uploads.append([path.name for path in paths])
            for path in paths:
                if self.failures and self.failures[0] == "500":
                    self.failures.pop(0)
                    raise RuntimeError("HTTP 500")
                if self.failures and self.failures[0] == "corrupt":
                    self.failures.pop(0)
                    self.put(path.name, path.read_bytes() + b"!")
                    continue
                self.put(path.name, path.read_bytes())
            return ""
        if arguments[:2] == ["release", "delete-asset"]:
            self.assets.pop(arguments[3])
            return ""
        raise AssertionError(arguments)


class SyncReleaseAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.release = write_release(self.tmp / "release")
        self.bundle = self.tmp / "bundle" / "recipe-library.tar"
        self.bundle.parent.mkdir()
        bundle_tool.build(self.release, self.bundle)
        self.wheel = self.tmp / "contracts-1.0.0-py3-none-any.whl"
        self.wheel.write_bytes(b"wheel built by this run")
        self.sources = {self.bundle.name: self.bundle, self.wheel.name: self.wheel}
        self.expected = sync_tool.expected_from_bundle(self.bundle)

    def converge(self, github: FakeGitHub) -> bool:
        """One publish run, as main() starts it from the release it finds."""

        wheel = self.sources.get(self.wheel.name)
        expected = self.expected | sync_tool.expected_wheel(
            github.assets, "v1.0.0", wheel
        )
        return sync_tool.converge(
            github,
            "owner/repo",
            "v1.0.0",
            expected,
            self.sources,
            sleep=lambda _seconds: None,
        )

    def test_uploads_the_bundle_and_the_wheel_into_an_empty_release(self) -> None:
        github = FakeGitHub()
        assert self.converge(github)
        assert github.uploads == [["recipe-library.tar", self.wheel.name]]
        assert set(github.assets) == {"recipe-library.tar", self.wheel.name}

    def test_a_rerun_never_replaces_the_versions_wheel(self) -> None:
        github = FakeGitHub()
        github.put("recipe-library.tar", self.bundle.read_bytes())
        github.put(self.wheel.name, b"wheel the first run uploaded")
        assert self.converge(github)
        assert github.uploads == []
        assert github.assets[self.wheel.name]["size"] == len(
            b"wheel the first run uploaded"
        )

    def test_a_run_killed_after_tagging_gets_its_wheel_next_run(self) -> None:
        github = FakeGitHub()
        github.put("recipe-library.tar", self.bundle.read_bytes())
        assert self.converge(github)
        assert github.uploads == [[self.wheel.name]]

    def test_a_wheel_this_run_uploads_corrupted_is_replaced(self) -> None:
        github = FakeGitHub(failures=["corrupt"])
        github.put("recipe-library.tar", self.bundle.read_bytes())
        assert self.converge(github)
        assert github.assets[self.wheel.name]["size"] == self.wheel.stat().st_size

    def test_a_missing_wheel_without_a_local_build_fails_visibly(self) -> None:
        github = FakeGitHub()
        self.sources = {self.bundle.name: self.bundle}
        assert not self.converge(github)
        assert "recipe-library.tar" in github.assets

    def test_a_server_error_mid_update_is_retried_until_it_converges(self) -> None:
        github = FakeGitHub(failures=["500"])
        assert self.converge(github)
        assert sync_tool.compare(
            self.expected | {self.wheel.name: sync_tool.Expected()},
            sync_tool.published_assets(github, "o/r", "v1"),
        ).clean

    def test_a_corrupted_asset_is_detected_by_digest_and_replaced(self) -> None:
        github = FakeGitHub(failures=["corrupt"])
        assert self.converge(github)

    def test_the_old_individual_assets_are_deleted_after_the_bundle_lands(self) -> None:
        github = FakeGitHub()
        for name, payload in FILES.items():
            github.put(name, payload)
        github.put("SHA256SUMS", b"old")
        github.put("SHA256SUMS.sigstore.json", b"old")
        assert self.converge(github)
        assert set(github.assets) == {"recipe-library.tar", self.wheel.name}
        assert github.uploads == [["recipe-library.tar", self.wheel.name]]

    def test_an_unconvergeable_release_fails_visibly(self) -> None:
        github = FakeGitHub(failures=["500"] * 20)
        assert not self.converge(github)

    def test_extra_assets_are_kept_while_the_bundle_cannot_be_uploaded(self) -> None:
        github = FakeGitHub(failures=["500"] * 20)
        github.put("alpha-recipe.tar.gz", b"old")
        self.converge(github)
        assert "recipe-library.tar" not in github.assets

    def test_a_missing_digest_falls_back_to_size_only_when_the_size_is_known(
        self,
    ) -> None:
        github = FakeGitHub()
        for name, payload in FILES.items():
            github.put(name, payload)
            github.assets[name]["digest"] = None
        difference = sync_tool.compare(
            {
                name: sync_tool.Expected(
                    hashlib.sha256(payload).hexdigest(), len(payload)
                )
                for name, payload in FILES.items()
            },
            github.assets,
        )
        assert difference.clean
        difference = sync_tool.compare(
            {
                name: sync_tool.Expected(hashlib.sha256(payload).hexdigest())
                for name, payload in FILES.items()
            },
            github.assets,
        )
        assert not difference.clean

    def test_check_needs_the_bundle_to_carry_the_manifest(self) -> None:
        manifest = self.release / "SHA256SUMS"
        assert sync_tool.bundle_matches_manifest(self.bundle, manifest)
        other = write_release(
            self.tmp / "other", {**FILES, "extra.tar.gz": b"\x1f\x8bextra"}
        )
        assert not sync_tool.bundle_matches_manifest(self.bundle, other / "SHA256SUMS")

    def test_check_rejects_a_bundle_with_a_tampered_member(self) -> None:
        tampered = self.tmp / "tampered.tar"
        with (
            tarfile.open(self.bundle) as source,
            tarfile.open(tampered, "w", format=tarfile.USTAR_FORMAT) as target,
        ):
            for member in source:
                payload = source.extractfile(member).read()  # type: ignore[union-attr]
                if member.name == "zeta-recipe.tar.gz":
                    payload = b"tampered"
                    member.size = len(payload)
                target.addfile(member, io.BytesIO(payload))
        assert not sync_tool.bundle_matches_manifest(
            tampered, self.release / "SHA256SUMS"
        )


if __name__ == "__main__":
    unittest.main()
