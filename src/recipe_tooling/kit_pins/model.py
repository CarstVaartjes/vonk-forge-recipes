"""Constants and records of the kit lock."""

from __future__ import annotations

import http.client
import re
from dataclasses import dataclass
from urllib.error import HTTPError

MANIFEST = "vendored-upstream.json"
ARCHIVE_ROUNDS = 3
LOCK = "kit-lock.json"
PUBLISHED_IMAGES = "policy/published-images.json"
SHA = re.compile(r"^[0-9a-f]{40}$")
ABBREVIATED_SHA = re.compile(r"^[0-9a-f]{7,40}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
KINDS = {"tag", "branch", "commit", "digest"}
# The immutable value each datasource resolves to, and the lock key that holds it.
IMMUTABLE_KEYS = {
    "git-ref": "commit",
    "hf-revision": "revision",
    "oci-image": "digest",
}
DATASOURCES = set(IMMUTABLE_KEYS)
GITHUB_REFERENCE = re.compile(
    r"^https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"/(?:tree|blob)/([0-9a-f]{40})(?:/.*)?$"
)
MAX_ARCHIVE_BYTES = 400_000_000
# Everything a download can raise, including mid-stream resets, timeouts and short reads.
FETCH_ERRORS = (OSError, http.client.HTTPException)
MAX_METADATA_LAYER = 1_000_000  # a layer that only adds a build-metadata file
MAX_SELECTION_ID = 64  # the contract bounds a selected file's id
OVERRIDE_VALUES = ("ref", "commit", "revision", "digest")


NOT_SERVED = {404, 410, 422}  # a definitive "not there", never an outage


class KitError(RuntimeError):
    """A declaration cannot be read or derived; the message names what to fix."""


def is_outage(error: BaseException) -> bool:
    """Whether ``error`` is a provider that could not be read (down, busy, timed out).

    A ``KitError`` is ours (a declaration or lock that is wrong) unless it wraps the read
    that failed.
    """
    chain: list[BaseException] = []
    link: BaseException | None = error
    while link is not None and link not in chain:
        chain.append(link)
        link = link.__cause__
    if any(isinstance(e, HTTPError) and e.code in NOT_SERVED for e in chain):
        return False  # the provider answered: the commit or image is not there
    if isinstance(error, KitError):
        return isinstance(error.__cause__, RuntimeError | OSError)
    return isinstance(error, RuntimeError | OSError)


@dataclass(frozen=True, slots=True)
class Declaration:
    name: str
    repo: str
    datasource: str
    file: str
    pattern: str
    override: dict[str, str] | None = None
    repo_pattern: str | None = None  # the repo is declared by the kit too
    binds: str | None = None  # hf-revision: the recipe model (``models[].id``) it pins

    @property
    def repo_in_pattern(self) -> bool:
        return "(?P<repo>" in self.pattern


@dataclass(frozen=True, slots=True)
class Pin:
    """One derived dependency, resolved to an immutable value."""

    name: str
    repo: str
    ref: str
    immutable: str  # a Git commit, a Hugging Face revision or an image digest
    date: str  # committer date of a commit or revision (YYYY-MM-DD); "" for an image
    kind: str = "tag"  # what ``ref`` names: tag, branch, commit or digest
    overridden: str | None = None  # the reason, when the pin is an override
    datasource: str = "git-ref"

    @property
    def version(self) -> str:
        return self.ref.removeprefix("v")

    @property
    def key(self) -> str:
        return IMMUTABLE_KEYS[self.datasource]
