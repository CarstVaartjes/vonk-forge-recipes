"""Constants and the blocking-file rules of the upstream refresh."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from recipe_tooling.paths import REPO_ROOT

ROOT = REPO_ROOT

SHA = re.compile(r"^[0-9a-f]{40}$")
#: Every refresher branch is ``<BRANCH_PREFIX><recipe-id>``.
BRANCH_PREFIX = "refresh/"
LABEL_NEEDS_REVIEW = "refresh:needs-review"
LABEL_MECHANICAL = "refresh:mechanical"
LABEL_MEMORY_ESTIMATE = "memory:estimate"
MEMORY_ESTIMATE_TITLE = "Memory estimate mismatches"
FAILED_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "STARTUP_FAILURE", "ERROR"}
# Full-suite verification takes minutes per recipe, so a run pushes (prepares) at most
# this many new or updated PRs; the rest wait for the next run, which the publish
# workflow chains. Arming is not bounded: every prepared PR is armed and GitHub merges
# each one when its required checks pass. Publication keeps up because publish.yml
# serialises on one concurrency group and every release is built from the newest main.
MAX_MECHANICAL_PER_RUN = 6
# A refresh chained from a finished publication is skipped when the last window already
# held this many successful refresh runs (a loop). A single recent run must never skip
# it: a run queued behind a long one always sees that one just finish.
CHAIN_WINDOW_MINUTES = 30
CHAIN_MAX_RUNS = 4
# A model revision that resizes the disk envelope by more than this needs a human.
MATERIAL_ENVELOPE_PERCENT = 1
# The proven recipes' exact bytes; a refresh that moves one rewrites its entry.
PROVEN_BYTES = "tests/fixtures/proven-recipe-bytes.json"
# What judges a refresh PR: a failed PR moves forward again when main changes these.
CHECK_PATHS = ("src/", "tools/", "tests/", ".github/")
MAX_BODY = 60000
VERSION_TAG = re.compile(r"^v?(\d+(?:\.\d+)*)$")
OWN_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Files whose change a script cannot judge: launch scripts, images, configuration,
# dependency lists and patches. READMEs, docs and licence texts are not judged: a recipe
# carries its own launch commands. Everything else in an upstream repository is not mirrored by a
# recipe, unless an adapter vendors it (that is checked byte for byte separately).
HARMLESS_DIRS = {
    ".github",
    "docs",
    "doc",
    "tests",
    "test",
    "assets",
    "images",
    "img",
    "bench",
    "benchmarks",
}
# A library the adapter pip-installs ships its examples and its own image recipes
# (docker/) for other users; the adapter runs neither.
LIBRARY_HARMLESS_DIRS = frozenset({"examples", "example", "docker"})
BLOCKING_SUFFIXES = {
    ".sh",
    ".bash",
    ".service",
    ".env",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".patch",
    ".diff",
    ".dockerfile",
    ".jinja",
}
BLOCKING_NAMES = re.compile(
    r"^(dockerfile.*|docker-compose.*|compose.*|makefile|"
    r"(requirements|constraints).*\.txt)$",
    re.IGNORECASE,
)
LAUNCH_SCRIPT = re.compile(
    r"^(launch|serve|server|run|start|entry|entrypoint|wrapper|deploy|vllm|sglang)"
    r"[-_a-z0-9]*\.py$",
    re.IGNORECASE,
)
LOCK_FILES = {"package-lock.json", "composer.lock", "tsconfig.json"}
# What a recipe never runs, wherever it sits: documentation, pictures and licence texts.
DOC_SUFFIXES = {".md", ".rst", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
DOC_NAMES = re.compile(
    r"^((licen[cs]e|notice|copying|authors|changelog)([.-].*)?|\.gitignore)$",
    re.IGNORECASE,
)


def is_blocking(path: str, harmless: frozenset[str] = frozenset()) -> bool:
    """Whether a changed upstream path can alter what a recipe launches."""
    pure = PurePosixPath(path)
    if pure.parts and pure.parts[0].lower() in HARMLESS_DIRS | harmless:
        return False
    name = pure.name
    if name in LOCK_FILES:
        return False
    return bool(
        pure.suffix.lower() in BLOCKING_SUFFIXES
        or BLOCKING_NAMES.match(name)
        or LAUNCH_SCRIPT.match(name)
    )


def _in_dirs(path: str, directories: frozenset[str]) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and parts[0].lower() in directories


def runs(path: str) -> bool:
    """Whether a changed upstream file can change what a recipe runs.

    Documentation, benchmarks, tests, CI, pictures and licence texts cannot: a move that
    changes only those is not a refresh, however the adapter uses the repository.
    """
    pure = PurePosixPath(path)
    if pure.parts and pure.parts[0].lower() in HARMLESS_DIRS:
        return False
    return not (pure.suffix.lower() in DOC_SUFFIXES or DOC_NAMES.match(pure.name))


def nothing_runs(
    comparison: Comparison,
    mirrored: set[str] | frozenset[str],
    harmless: frozenset[str] = frozenset(),
) -> bool:
    """Whether a complete comparison changed only files no recipe runs."""
    names = {f["filename"] for f in comparison.files} | {
        f["previous_filename"] for f in comparison.files if "previous_filename" in f
    }
    return not comparison.truncated and not any(
        (runs(name) and not _in_dirs(name, harmless)) or name in mirrored
        for name in names
    )


# ---------------------------------------------------------------------------
# The pinning rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Target:
    """What the pinning rule selects for one upstream repository."""

    provider: str
    repo: str
    sha: str
    date: str
    ref: str | None  # tag for a published version, None when following the head
    version: str | None  # upstream's version spelling (no leading "v")
    url: str
    notes: str | None = None
    access_status: str | None = None
    branch: str | None = None  # non-default branch the pin follows, when it does
    committed_at: str | None = None  # the commit's own timestamp (ISO 8601), when known


# ---------------------------------------------------------------------------
# Items: one drifted pin
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Item:
    kind: str  # source | adapter | model
    provider: str
    repo: str
    current: str
    target: Target
    model_slug: str | None = None
    comparison: Comparison | None = None  # pin -> newest, GitHub items only
    noop: bool = False  # assessed as not worth a move (set while assessing)
    kit_pin: Any = (
        None  # a kit-declared dependency's pin: the target is the kit's, not the newest
    )
    image: tuple[str, str] | None = None  # (repository, digest) of a followed image
    # The image's build commit is not on GitHub: the move is prepared with this
    # provenance mark (vendored-upstream.json source.unverified) and goes to review.
    unverified: str | None = None
    unmark: bool = False  # the marked commit is on GitHub now: only the mark goes
    # Moving off a never-published pin to a published commit: nothing to compare with.
    uncompared: bool = False

    @property
    def label(self) -> str:
        return f"{self.kind} {self.provider}:{self.repo}"


@dataclass(slots=True)
class Assessment:
    recipe_id: str
    items: list[Item] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # Why a prepared move waits with its build-context group (shown in the issue, never a
    # reason: it leaves prepared/mechanical, and so the group's open pull request, alone).
    group_outcome: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # newer release lines: never a move
    # Why a prepared move still needs a person (an image whose build commit GitHub does
    # not serve): the edits are made, the pull request is not armed.
    held: list[str] = field(default_factory=list)
    pull_request: int | None = None  # the prepared pull request, when there is one
    edits: dict[str, bytes | None] = field(default_factory=dict)
    version: tuple[str, str] | None = None  # (version, released_at) after refresh
    members: list[str] = field(default_factory=list)  # recipes refreshed together
    # Pins a hold or a declined revision keeps where they are (labels), and whether the
    # move raised any role peak: the PR body claims only what happened.
    declined: list[str] = field(default_factory=list)
    peaks_raised: bool = False

    @property
    def label(self) -> str:
        return ", ".join(self.members) or self.recipe_id

    @property
    def drifted(self) -> bool:
        return bool(self.items or self.reasons)

    @property
    def mechanical(self) -> bool:
        return (
            self.drifted and not self.reasons and not self.unreachable and not self.held
        )

    @property
    def for_review(self) -> bool:
        """A move prepared as a pull request that a person merges (not armed)."""
        return bool(self.held) and not self.reasons and not self.unreachable

    @property
    def prepared(self) -> bool:
        return self.mechanical or self.for_review

    def fingerprint(self) -> str:
        text = "|".join(sorted(f"{i.kind}:{i.repo}:{i.target.sha}" for i in self.items))
        return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparison:
    status: str
    commits: list[dict[str, Any]]
    files: list[dict[str, Any]]
    truncated: bool
    url: str
    rewritten: str = ""  # the new head when GitHub finds no common ancestor


METADATA_ONLY = {"metadata"}
