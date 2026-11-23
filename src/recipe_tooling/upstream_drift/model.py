"""Constants and records of the upstream drift check."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from vonk_forge_contracts.model import GitHubReleaseSource

from recipe_tooling.paths import REPO_ROOT

ROOT = REPO_ROOT

DEFAULT_MANIFEST = ROOT / "policy/upstream-watch.json"
SHA = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
ENTITY_DIRECTORIES = {
    "models": "model",
}
POLICIES = {"default-branch", "latest-release", "manual"}
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
# Creator signals. The window is far wider than the daily cadence, so missed runs
# still see every new repository without any stored "last scan" state.
NEW_REPO_WINDOW = timedelta(days=14)
SIGNAL_WORKERS = 4
NOTICE_HEAD_LINES = 60
# Supersession banners seen so far (Mia's AI Lab 2026-11-04) sit in the first
# README block as "superseded" next to a link to the successor repository.
NOTICE_WORDS = re.compile(
    r"supersed|deprecat|moved to|use the new|replaced by|no longer maintained",
    re.IGNORECASE,
)
# owner/name plus the next path segment, so issue and pull links can be refused.
GITHUB_REPO_LINK = re.compile(
    r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:/([A-Za-z_-]+))?"
    r"(?=[\s)\]>\"'#?/]|$)"
)
# github.com/<first segment> values that are not repository owners.
NON_OWNER_SEGMENTS = frozenset(
    {
        "apps", "collections", "features", "login", "marketplace", "notifications",
        "orgs", "security", "settings", "sponsors", "topics", "user-attachments",
    }
)  # fmt: skip
NON_REPO_PAGES = frozenset({"pull", "pulls", "issues", "discussions"})
MAX_LISTING_PAGES = 5
LISTING_PAGE_SIZE = 100
LABELS = {"supersession": "upstream-superseded", "new-repo": "upstream-new-repo"}
DEDUPE_MARKER = "<!-- dedupe-key: {key} -->"


class DriftInputError(RuntimeError):
    """The local watch configuration or a catalog source is invalid."""


class RemoteNotFound(RuntimeError):
    """The provider reports that a repository, revision, or release is absent."""


@dataclass(frozen=True, slots=True)
class SourceLocation:
    provider: str
    repository: str
    embedded_revision: str | None = None


@dataclass(frozen=True, slots=True)
class Watch:
    entity: str
    source_path: str
    provider: str
    repository: str
    # Git commit for tree sources; numeric release ID for release-asset sources.
    pinned_revision: str
    pinned_version: str | None
    policy: str
    ref: str | None
    release_source: GitHubReleaseSource | None = None
    # Catalog facts already validated while discovering watches, so the creator
    # signals never read a document again. family: a model's family slug;
    # models: the model entities a recipe uses.
    family: str | None = None
    models: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Observation:
    entity: str
    source_path: str
    provider: str
    repository: str
    policy: str
    ref: str | None
    pinned_revision: str
    pinned_version: str | None
    observed_revision: str | None
    observed_version: str | None
    observed_at: str | None
    upstream_url: str | None
    status: str
    detail: str | None


class _Finding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CreatorFinding(_Finding):
    """One machine-actionable creator signal; a scheduled agent acts on it.

    supersession: source_repo's README head names linked_repo as its successor;
    recipe_slugs are the recipes pinned to source_repo.
    new-repo: linked_repo is a recent repository of a followed creator that names
    a carried model family; recipe_slugs are the recipes whose model is in it.
    """

    kind: Literal["supersession", "new-repo"]
    label: Literal["upstream-superseded", "upstream-new-repo"]
    dedupe_key: str
    source_repo: str | None
    source_commit: str | None
    linked_repo: str
    linked_commit: str | None
    recipe_slugs: list[str]
    matched_family: str | None
    detail: str

    @model_validator(mode="before")
    @classmethod
    def _label_from_kind(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and "label" not in data
            and data.get("kind") in LABELS
        ):
            return {**data, "label": LABELS[data["kind"]]}
        return data

    @model_validator(mode="after")
    def _label_matches_kind(self) -> CreatorFinding:
        if LABELS[self.kind] != self.label:
            raise ValueError("label does not match kind")
        return self

    @field_validator("detail")
    @classmethod
    def _no_fence(cls, value: str) -> str:
        # The record is embedded in a fenced issue body.
        return value.replace("`", "'")


class SignalError(_Finding):
    subject: str
    detail: str


class CreatorSignals(_Finding):
    findings: list[CreatorFinding] = []
    signal_errors: list[SignalError] = []


class Creator(_Finding):
    name: str
    url: str
    aliases: list[str]
    focus: str


class CreatorsFile(_Finding):
    comment: str = Field(alias="_comment")
    creators: list[Creator]
