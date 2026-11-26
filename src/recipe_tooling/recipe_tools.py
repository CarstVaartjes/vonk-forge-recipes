"""The signed recipe-tools bundle of one Vonk Forge channel: fetch, verify, extract.

Recipes CI validates the library with the platform each channel ships, not with
a pinned forge commit. Each forge release generation publishes, under
``https://install.vonkforge.ai/artifacts/<channel>/``:

* ``recipe-tools-<sha256>.tar.gz``, named by its own digest and written once;
* ``recipe-tools.json``, the pointer ``{"bundle", "ref", "sha256", "source_sha"}``,
  written last (forge ``docs/runbooks/platform-release-publication.md``).

This is a security edge, so it fails closed: the pointer's shape, the bundle's
sha256 and its keyless attestation by forge's release workflow at the pointer's
source commit and ref are all checked before anything is extracted. Bytes are
checked once, here; a cached bundle is not hashed or verified again.

Newest wins. A reachable pointer is authoritative: a bundle that is missing or
does not verify fails the run, and the next run fetches again; there is no
fallback to an older bundle. Only when the channel host does not answer at all
does the run use the newest bundle an earlier run already verified (the CI cache
holds one per channel), so a forge outage does not stop recipe publication. A
stable pointer that does not exist yet (404) skips that channel; it is the only
skip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError

from recipe_tooling.upstream_http import fetch as http_fetch

BASE_URL = "https://install.vonkforge.ai/artifacts"
FORGE = "CarstVaartjes/vonk-forge"
SIGNER_WORKFLOW = f"{FORGE}/.github/workflows/installer-publication.yml"
POINTER = "recipe-tools.json"
CHANNELS = ("dev", "stable")
# Until the first stable release carries a bundle, its pointer is absent.
ABSENT_SKIPS = frozenset({"stable"})

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TAG_REF = re.compile(r"^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$")


class Absent(Exception):
    """The channel answered 404 for this file."""


class Unreachable(Exception):
    """The channel host did not answer (network failure or server error)."""


class Refused(Exception):
    """The pointer or the bundle failed a check; nothing was extracted or cached."""


@dataclass(frozen=True)
class Pointer:
    channel: str
    source_sha: str
    ref: str
    bundle: str
    sha256: str


@dataclass(frozen=True)
class Obtained:
    """``fresh``: downloaded and verified now. ``cached``: the pointer names the
    cached bundle. ``offline``: the host did not answer, the cached bundle is used.
    ``absent``: the channel has no pointer yet (stable only); nothing to validate."""

    state: Literal["fresh", "cached", "offline", "absent"]
    pointer: Pointer | None


Fetch = Callable[[str], bytes]
Verify = Callable[[Path, Pointer], None]


def parse_pointer(channel: str, raw: bytes) -> Pointer:
    """The pointer, checked: exact digests, the digest-named bundle, the channel's ref."""

    try:
        document = json.loads(raw)
        fields = {k: document[k] for k in ("source_sha", "ref", "bundle", "sha256")}
    except (ValueError, TypeError, KeyError) as error:
        raise Refused(f"{channel} pointer is not readable: {error}") from error
    if not all(isinstance(value, str) for value in fields.values()):
        raise Refused(f"{channel} pointer fields must be strings")
    pointer = Pointer(channel=channel, **fields)
    expected_ref = (
        pointer.ref == "refs/heads/main"
        if channel == "dev"
        else bool(_TAG_REF.fullmatch(pointer.ref))
    )
    if (
        not _SHA1.fullmatch(pointer.source_sha)
        or not _SHA256.fullmatch(pointer.sha256)
        or pointer.bundle != f"recipe-tools-{pointer.sha256}.tar.gz"
        or not expected_ref
    ):
        raise Refused(f"{channel} pointer is malformed: {fields}")
    return pointer


def _cached(channel: str, cache: Path) -> Pointer | None:
    """The bundle an earlier run verified, if its pointer and bytes are both there."""

    try:
        pointer = parse_pointer(channel, (cache / POINTER).read_bytes())
    except (OSError, Refused):
        return None
    return pointer if (cache / pointer.bundle).is_file() else None


def obtain(channel: str, fetch: Fetch, verify: Verify, cache: Path) -> Obtained:
    """Make ``cache`` hold the newest verified bundle of ``channel``."""

    try:
        raw = fetch(POINTER)
    except Absent:
        if channel in ABSENT_SKIPS:
            return Obtained("absent", None)
        raise
    except Unreachable:
        cached = _cached(channel, cache)
        if cached is None:
            raise
        return Obtained("offline", cached)
    pointer = parse_pointer(channel, raw)
    if _cached(channel, cache) == pointer:
        return Obtained("cached", pointer)
    content = fetch(pointer.bundle)
    if hashlib.sha256(content).hexdigest() != pointer.sha256:
        raise Refused(f"{pointer.bundle} does not match the pointer's sha256")
    # Verified in a staging file; the cache is replaced only after it passes,
    # and the pointer is written last, so a cut-off run leaves no usable entry.
    staging = cache.with_name(cache.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    (staging / pointer.bundle).write_bytes(content)
    try:
        verify(staging / pointer.bundle, pointer)
    except Exception:
        shutil.rmtree(staging)
        raise
    shutil.rmtree(cache, ignore_errors=True)
    staging.rename(cache)
    (cache / POINTER).write_bytes(raw)
    return Obtained("fresh", pointer)


def extract(cache: Path, pointer: Pointer, dest: Path) -> None:
    """Unpack the cached bundle into an empty ``dest``."""

    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(cache / pointer.bundle, "r:gz") as archive:
        archive.extractall(dest, filter="data")


def channel_fetch(channel: str, base_url: str = BASE_URL) -> Fetch:
    """Read one file of the channel over HTTPS, with ``upstream_http``'s bounded retry."""

    def read(name: str) -> bytes:
        # The host refuses urllib's default User-Agent (403).
        request = urllib.request.Request(
            f"{base_url}/{channel}/{name}", headers={"User-Agent": "vonk-forge-recipes"}
        )
        try:
            return http_fetch(request, lambda response: response.read(), timeout=60)
        except HTTPError as error:
            if error.code == 404:
                raise Absent(name) from error
            if error.code >= 500:  # a proxy's own 52x as well as the origin's
                raise Unreachable(f"{base_url}/{channel}/{name}: {error}") from error
            raise
        except RuntimeError as error:  # transient failures past the retry bound
            raise Unreachable(f"{base_url}/{channel}/{name}: {error}") from error

    return read


def attestation_verify(path: Path, pointer: Pointer) -> None:
    """``gh attestation verify``: signed by forge's release workflow at the pointer's commit."""

    subprocess.run(
        [
            "gh",
            "attestation",
            "verify",
            str(path),
            "--repo",
            FORGE,
            "--signer-workflow",
            SIGNER_WORKFLOW,
            "--source-digest",
            pointer.source_sha,
            "--source-ref",
            pointer.ref,
        ],
        check=True,
    )


SUMMARY = {
    "fresh": "Validated with the {channel} recipe-tools bundle of `{source}` (verified now).",
    "cached": "Validated with the {channel} recipe-tools bundle of `{source}` (verified earlier).",
    "offline": (
        "The {channel} channel did not answer; validated with the newest bundle "
        "verified earlier, of `{source}`."
    ),
    "absent": "The {channel} channel has published no recipe-tools bundle yet; skipped.",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=CHANNELS, required=True)
    parser.add_argument(
        "--cache", type=Path, required=True, help="this channel's cache"
    )
    parser.add_argument("--dest", type=Path, required=True, help="where to extract it")
    args = parser.parse_args(argv)
    got = obtain(
        args.channel, channel_fetch(args.channel), attestation_verify, args.cache
    )
    if got.pointer is not None:
        extract(args.cache, got.pointer, args.dest)
    # Step outputs on stdout (the workflow appends them to $GITHUB_OUTPUT).
    print(f"state={got.state}")
    print(f"sha256={got.pointer.sha256 if got.pointer else ''}")
    line = SUMMARY[got.state].format(
        channel=args.channel, source=got.pointer.source_sha if got.pointer else ""
    )
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return 0
