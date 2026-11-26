"""The channel bundle: verified once at ingress, newest wins, an outage uses the cache."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import pytest
from repo_root import ROOT

from recipe_tooling import recipe_tools
from recipe_tooling.recipe_tools import Absent, Pointer, Refused, Unreachable


def _bundle(text: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = text.encode()
        member = tarfile.TarInfo("platform-features.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


class FakeChannel:
    """One channel directory: its files, or an outage."""

    def __init__(self, ref: str = "refs/heads/main") -> None:
        self.files: dict[str, bytes] = {}
        self.down = False
        self.ref = ref

    def publish(self, text: str, source: str = "a" * 40) -> str:
        content = _bundle(text)
        sha256 = hashlib.sha256(content).hexdigest()
        name = f"recipe-tools-{sha256}.tar.gz"
        self.files[name] = content
        pointer = {
            "bundle": name,
            "ref": self.ref,
            "sha256": sha256,
            "source_sha": source,
        }
        self.files["recipe-tools.json"] = json.dumps(pointer).encode()
        return sha256

    def fetch(self, name: str) -> bytes:
        if self.down:
            raise Unreachable(name)
        if name not in self.files:
            raise Absent(name)
        return self.files[name]


class Verifier:
    def __init__(self, refuse: bool = False) -> None:
        self.calls: list[str] = []
        self.refuse = refuse

    def __call__(self, path: Path, pointer: Pointer) -> None:
        self.calls.append(pointer.sha256)
        if self.refuse:
            raise RuntimeError("attestation refused")


def _features(cache: Path, pointer: Pointer, dest: Path) -> str:
    recipe_tools.extract(cache, pointer, dest)
    return (dest / "platform-features.json").read_text()


def test_verified_once_then_cached_and_newest_replaces_it(tmp_path: Path) -> None:
    channel, verify, cache = FakeChannel(), Verifier(), tmp_path / "cache"
    first = channel.publish("one")
    got = recipe_tools.obtain("dev", channel.fetch, verify, cache)
    assert (got.state, verify.calls) == ("fresh", [first])
    again = recipe_tools.obtain("dev", channel.fetch, verify, cache)
    assert (again.state, verify.calls) == ("cached", [first])
    second = channel.publish("two", source="b" * 40)
    newest = recipe_tools.obtain("dev", channel.fetch, verify, cache)
    assert (newest.state, verify.calls) == ("fresh", [first, second])
    assert newest.pointer is not None
    assert _features(cache, newest.pointer, tmp_path / "x") == "two"
    assert sorted(p.name for p in cache.iterdir()) == [
        f"recipe-tools-{second}.tar.gz",
        "recipe-tools.json",
    ]


def test_a_refused_bundle_is_neither_cached_nor_used_and_the_next_run_retries(
    tmp_path: Path,
) -> None:
    channel, cache = FakeChannel(), tmp_path / "cache"
    channel.publish("old")
    recipe_tools.obtain("dev", channel.fetch, Verifier(), cache)
    channel.publish("new", source="b" * 40)
    with pytest.raises(RuntimeError, match="attestation refused"):
        recipe_tools.obtain("dev", channel.fetch, Verifier(refuse=True), cache)
    retry = Verifier()
    assert recipe_tools.obtain("dev", channel.fetch, retry, cache).state == "fresh"
    assert len(retry.calls) == 1


def test_a_digest_mismatch_is_refused_before_verification(tmp_path: Path) -> None:
    channel, verify = FakeChannel(), Verifier()
    sha256 = channel.publish("x")
    channel.files[f"recipe-tools-{sha256}.tar.gz"] = _bundle("tampered")
    with pytest.raises(Refused):
        recipe_tools.obtain("dev", channel.fetch, verify, tmp_path / "cache")
    assert verify.calls == []


def test_an_outage_uses_the_verified_cache_and_fails_without_one(
    tmp_path: Path,
) -> None:
    channel, cache = FakeChannel(), tmp_path / "cache"
    channel.publish("known")
    recipe_tools.obtain("dev", channel.fetch, Verifier(), cache)
    channel.down = True
    verify = Verifier()
    offline = recipe_tools.obtain("dev", channel.fetch, verify, cache)
    assert (offline.state, verify.calls) == ("offline", [])
    with pytest.raises(Unreachable):
        recipe_tools.obtain("dev", channel.fetch, verify, tmp_path / "empty")


def test_only_a_missing_stable_pointer_skips(tmp_path: Path) -> None:
    empty = FakeChannel()
    got = recipe_tools.obtain("stable", empty.fetch, Verifier(), tmp_path / "s")
    assert (got.state, got.pointer) == ("absent", None)
    with pytest.raises(Absent):
        recipe_tools.obtain("dev", empty.fetch, Verifier(), tmp_path / "d")


@pytest.mark.parametrize(
    ("channel", "ref"),
    [
        ("dev", "refs/heads/feature"),
        ("stable", "refs/heads/main"),
        ("stable", "refs/tags/v1.*"),
    ],
)
def test_the_pointer_ref_must_be_the_channels(
    tmp_path: Path, channel: str, ref: str
) -> None:
    fake = FakeChannel(ref=ref)
    fake.publish("x")
    with pytest.raises(Refused):
        recipe_tools.obtain(channel, fake.fetch, Verifier(), tmp_path / "cache")


def test_no_workflow_pins_a_forge_commit() -> None:
    """Platform validation comes from the channel bundle, never a pinned forge workflow."""

    pinned = re.compile(
        r"vonk-forge/\.github/workflows/\S*@|platform_ref:\s*[0-9a-f]{40}"
        r"|repository:\s*CarstVaartjes/vonk-forge\s*$",
        re.MULTILINE,
    )
    workflows = sorted((ROOT / ".github/workflows").glob("*.yml"))
    assert workflows
    assert [p.name for p in workflows if pinned.search(p.read_text("utf-8"))] == []
