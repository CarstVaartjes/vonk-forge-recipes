"""Observing the upstream heads of the watches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Any, Protocol
from urllib.parse import quote

from recipe_tooling.upstream_clients import (
    ProviderDecodeError,
    ProviderHttpError,
    ProviderJson,
    ProviderTooLarge,
)
from recipe_tooling.upstream_drift.model import (
    MAX_RESPONSE_BYTES,
    SHA,
    CreatorSignals,
    Observation,
    RemoteNotFound,
    Watch,
)
from recipe_tooling.upstream_http import UpstreamBusy, open_once


class JsonClient(Protocol):
    """What the head observations need of a client: one provider JSON read."""

    def json(self, url: str, provider: str, /) -> object: ...


class ProviderClient:
    def __init__(
        self,
        *,
        github_token: str | None = None,
        huggingface_token: str | None = None,
        opener: Callable[..., AbstractContextManager[Any]] = open_once,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.github_token = github_token
        self.huggingface_token = huggingface_token
        self.opener = opener
        self.client = ProviderJson(
            user_agent="vonk-forge-upstream-drift/1",
            timeout=20,
            tokens={"github": github_token, "huggingface": huggingface_token},
            max_bytes=MAX_RESPONSE_BYTES,
            github_api_version="2022-11-28",
            opener=opener,
            sleep=sleep,
        )

    def json(self, url: str, provider: str) -> object:
        token = self.github_token if provider == "github" else self.huggingface_token
        try:
            return self.client.get(url, provider)
        except ProviderHttpError as error:
            # Hugging Face answers an anonymous request for a deleted or
            # private repository with 401, not 404 (gated ones still list).
            if error.code == 404 or (
                error.code == 401 and provider == "huggingface" and not token
            ):
                raise RemoteNotFound(url) from error.error
            raise RuntimeError(f"provider returned HTTP {error.code}") from error.error
        except ProviderTooLarge as error:
            raise RuntimeError(str(error)) from error
        except ProviderDecodeError as error:
            raise RuntimeError(f"provider request failed: {error}") from error


def _required_string(document: object, key: str, context: str) -> str:
    value = document.get(key) if isinstance(document, dict) else None
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"provider response is missing {context}.{key}")
    return value


def _github_commit(
    client: JsonClient, repository: str, ref: str
) -> tuple[str, str | None, str]:
    encoded = quote(ref, safe="")
    document = client.json(
        f"https://api.github.com/repos/{repository}/commits/{encoded}", "github"
    )
    revision = _required_string(document, "sha", "commit")
    url = _required_string(document, "html_url", "commit")
    commit = document.get("commit") if isinstance(document, dict) else None
    committer = commit.get("committer") if isinstance(commit, dict) else None
    observed_at = committer.get("date") if isinstance(committer, dict) else None
    return revision, observed_at if isinstance(observed_at, str) else None, url


def _pinned_exists(client: ProviderClient, watch: Watch) -> bool:
    try:
        if watch.release_source is not None:
            release = client.json(
                f"https://api.github.com/repos/{watch.repository}/releases/{watch.release_source.release_id}",
                "github",
            )
            assets = release.get("assets") if isinstance(release, dict) else None
            if not isinstance(assets, list):
                raise RuntimeError("provider response is missing release.assets")
            asset_ids = {asset.get("id") for asset in assets if isinstance(asset, dict)}
            return all(
                asset.asset_id in asset_ids for asset in watch.release_source.assets
            )
        if watch.provider == "github":
            _github_commit(client, watch.repository, watch.pinned_revision)
        elif watch.provider == "huggingface":
            repository = quote(watch.repository, safe="/")
            client.json(
                f"https://huggingface.co/api/models/{repository}/revision/{watch.pinned_revision}",
                "huggingface",
            )
        else:
            return True
    except RemoteNotFound:
        return False
    return True


def observe(watch: Watch, client: ProviderClient) -> Observation:
    common = {
        "entity": watch.entity,
        "source_path": watch.source_path,
        "provider": watch.provider,
        "repository": watch.repository,
        "policy": watch.policy,
        "ref": watch.ref,
        "pinned_revision": watch.pinned_revision,
        "pinned_version": watch.pinned_version,
    }
    if watch.policy == "manual" or watch.provider not in {"github", "huggingface"}:
        return Observation(
            **common,
            observed_revision=None,
            observed_version=None,
            observed_at=None,
            upstream_url=None,
            status="untracked",
            detail="no automated upstream policy",
        )
    try:
        observed_version: str | None = None
        if watch.release_source is not None:
            if not _pinned_exists(client, watch):
                raise RemoteNotFound("pinned release or selected assets are missing")
            release = client.json(
                f"https://api.github.com/repos/{watch.repository}/releases/latest",
                "github",
            )
            if not isinstance(release, dict):
                raise RuntimeError("provider response is not a release object")
            release_id = release.get("id")
            if type(release_id) is not int or release_id <= 0:
                raise RuntimeError("provider response is missing release.id")
            revision = str(release_id)
            observed_version = _required_string(release, "tag_name", "release")
            upstream_url = _required_string(release, "html_url", "release")
            observed_at_value = release.get("published_at")
            observed_at = (
                observed_at_value if isinstance(observed_at_value, str) else None
            )
        elif watch.provider == "github":
            if watch.policy == "latest-release":
                release = client.json(
                    f"https://api.github.com/repos/{watch.repository}/releases/latest",
                    "github",
                )
                ref = _required_string(release, "tag_name", "release")
                observed_version = ref
                revision, observed_at, upstream_url = _github_commit(
                    client, watch.repository, ref
                )
            else:
                ref = watch.ref
                if ref is None:
                    repository = client.json(
                        f"https://api.github.com/repos/{watch.repository}", "github"
                    )
                    ref = _required_string(repository, "default_branch", "repository")
                revision, observed_at, upstream_url = _github_commit(
                    client, watch.repository, ref
                )
        else:
            repository = quote(watch.repository, safe="/")
            suffix = f"/revision/{quote(watch.ref, safe='')}" if watch.ref else ""
            document = client.json(
                f"https://huggingface.co/api/models/{repository}{suffix}",
                "huggingface",
            )
            revision = _required_string(document, "sha", "model")
            observed_at_value = (
                document.get("lastModified") if isinstance(document, dict) else None
            )
            observed_at = (
                observed_at_value if isinstance(observed_at_value, str) else None
            )
            upstream_url = f"https://huggingface.co/{watch.repository}/tree/{revision}"
        if watch.release_source is None and not SHA.fullmatch(revision):
            raise RuntimeError("provider returned a non-Git revision")
        status = "current" if revision == watch.pinned_revision else "advanced"
        detail = None
        if status == "advanced" and not _pinned_exists(client, watch):
            status = "missing"
            detail = "pinned revision is no longer resolvable"
        return Observation(
            **common,
            observed_revision=revision,
            observed_version=observed_version,
            observed_at=observed_at,
            upstream_url=upstream_url,
            status=status,
            detail=detail,
        )
    except RemoteNotFound:
        return Observation(
            **common,
            observed_revision=None,
            observed_version=None,
            observed_at=None,
            upstream_url=None,
            status="missing",
            detail="upstream repository, release, or revision was not found",
        )
    except RuntimeError as error:
        # A rate-limited or unreachable provider is "not checked this run", not a fault.
        return Observation(
            **common,
            observed_revision=None,
            observed_version=None,
            observed_at=None,
            upstream_url=None,
            status="unchecked" if isinstance(error, UpstreamBusy) else "error",
            detail=str(error)[:256],
        )


def report(
    observations: list[Observation],
    checked_at: str,
    embedded: dict[str, object] | None = None,
    signals: CreatorSignals | None = None,
) -> dict[str, object]:
    counts = Counter(observation.status for observation in observations)
    return {
        "schema_version": 2,
        "checked_at": checked_at,
        "counts": {
            status: counts.get(status, 0)
            for status in (
                "current",
                "advanced",
                "missing",
                "untracked",
                "unchecked",
                "error",
            )
        },
        "observations": [asdict(observation) for observation in observations],
        "embedded_inputs": embedded
        or {"coverage": {"scope": "not collected"}, "recipes": []},
        "creator_signals": (signals or CreatorSignals()).model_dump(mode="json"),
    }
