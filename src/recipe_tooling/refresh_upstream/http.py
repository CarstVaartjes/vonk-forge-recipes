"""The HTTP client of the upstream refresh."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from recipe_tooling.kit_pins import upstream as kit_providers
from recipe_tooling.upstream_http import fetch

# ---------------------------------------------------------------------------
# HTTP with conditional requests
# ---------------------------------------------------------------------------


class NotFound(RuntimeError):
    """HTTP 404; ``message`` is the provider's own explanation when it gave one."""

    def __init__(self, url: str, message: str = "") -> None:
        super().__init__(url)
        self.message = message


def _error_message(error: urllib.error.HTTPError) -> str:
    """The ``message`` of a provider's JSON error body ('' when there is none)."""
    try:
        body = json.loads(error.read())
        return str(body.get("message", "")) if isinstance(body, dict) else ""
    except (OSError, ValueError):
        return ""


class AccessRestricted(RuntimeError):
    """An anonymous provider request requires access the catalog does not bind."""


class Http:
    """GET JSON with an ETag cache: a 304 answer costs no API quota."""

    def __init__(
        self,
        *,
        github_token: str | None,
        hf_token: str | None,
        cache_path: Path | None,
    ) -> None:
        self.tokens = {"github": github_token, "huggingface": hf_token}
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.memo: dict[str, object] = {}
        self.kit_source: Any = None  # kit_upstream: one per run
        self.requests = 0
        self.not_modified = 0
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path and cache_path.is_file():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def json(self, url: str, provider: str, *, anonymous: bool = False) -> Any:
        cache_key = f"{url}#anonymous" if anonymous else url
        with self.lock:
            if cache_key in self.memo:
                return self.memo[cache_key]
            cached = self.cache.get(cache_key)
        headers = {
            "Accept": "application/vnd.github+json"
            if provider == "github"
            else "application/json",
            "User-Agent": "vonk-forge-refresh-upstream/1",
        }
        token = None if anonymous else self.tokens.get(provider)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if cached:
            headers["If-None-Match"] = cached["etag"]
        request = urllib.request.Request(url, headers=headers)

        try:
            payload, etag = fetch(
                request,
                lambda response: (response.read(), response.headers.get("ETag")),
                timeout=30,
            )
        except urllib.error.HTTPError as error:
            if anonymous and error.code in {401, 403}:
                raise AccessRestricted(
                    f"anonymous provider request returned HTTP {error.code}"
                ) from error
            if error.code == 304 and cached:
                with self.lock:
                    self.requests += 1
                    self.not_modified += 1
                value = cached["body"]
            elif error.code == 404:
                raise NotFound(url, _error_message(error)) from error
            else:
                raise RuntimeError(f"HTTP {error.code} for {url}") from error
        else:
            with self.lock:
                self.requests += 1
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"request failed for {url}: {error}") from error
            if etag and len(payload) < 2_000_000:
                with self.lock:
                    self.cache[cache_key] = {"etag": etag, "body": value}
        with self.lock:
            self.memo[cache_key] = value
        return value

    def github(self, path: str) -> Any:
        return self.json(f"https://api.github.com/{path}", "github")


def kit_upstream(http: Http) -> Any:
    """Where a kit's declared dependencies are resolved: GitHub, Hugging Face, registries.

    One per run (kept on ``http``), so a registry answer is fetched once for every recipe.
    """
    found = getattr(http, "kit_source", None)
    if found is None:
        found = kit_providers.Upstream(
            kit_providers.GitHub(http.github),
            kit_providers.HuggingFace(
                lambda path: http.json(
                    f"https://huggingface.co/api/{path}", "huggingface", anonymous=True
                )
            ),
            kit_providers.Registry(kit_providers.fetch),
        )
        http.kit_source = found
    return found
