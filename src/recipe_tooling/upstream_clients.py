"""One JSON client for the provider APIs (GitHub, Hugging Face) over ``upstream_http.fetch``.

The refresh and the drift check each read provider JSON with the same request shape,
token handling, bounded read and decoding; they differ only in the user agent, the
timeout, the size bound and whether an ETag cache is kept. Those are arguments here, so
each caller passes the values it always used. A failed answer raises ``ProviderHttpError``
or ``ProviderDecodeError`` and the caller words it as it always did.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from recipe_tooling.upstream_http import fetch


class ProviderHttpError(RuntimeError):
    """The provider answered with an HTTP error status."""

    def __init__(self, url: str, error: urllib.error.HTTPError) -> None:
        super().__init__(f"HTTP {error.code} for {url}")
        self.url = url
        self.code = error.code
        self.error = error


class ProviderDecodeError(RuntimeError):
    """The provider's answer is not JSON."""


class ProviderTooLarge(ProviderDecodeError):
    """The provider's answer exceeds the size bound."""


class ProviderJson:
    """GET JSON with optional ETag revalidation, a size bound and a per-run memo."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float,
        tokens: dict[str, str | None] | None = None,
        max_bytes: int | None = None,
        etag_cache: dict[str, dict[str, Any]] | None = None,
        github_api_version: str | None = None,
        opener: Callable[..., AbstractContextManager[Any]] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.tokens = tokens or {}
        self.max_bytes = max_bytes
        self.etag_cache = etag_cache
        self.github_api_version = github_api_version
        self.opener = opener
        self.sleep = sleep
        self.lock = threading.Lock()
        self.memo: dict[str, object] = {}
        self.requests = 0
        self.not_modified = 0

    def headers(self, provider: str, token: str | None) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json"
            if provider == "github"
            else "application/json",
            "User-Agent": self.user_agent,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if provider == "github" and self.github_api_version:
            headers["X-GitHub-Api-Version"] = self.github_api_version
        return headers

    def get(self, url: str, provider: str, *, anonymous: bool = False) -> Any:
        key = f"{url}#anonymous" if anonymous else url
        with self.lock:
            if key in self.memo:
                return self.memo[key]
            cached = self.etag_cache.get(key) if self.etag_cache is not None else None
        token = None if anonymous else self.tokens.get(provider)
        headers = self.headers(provider, token)
        if cached:
            headers["If-None-Match"] = cached["etag"]
        request = urllib.request.Request(url, headers=headers)
        limit = self.max_bytes

        def read(response: Any) -> tuple[bytes, str | None]:
            payload = response.read() if limit is None else response.read(limit + 1)
            etag = response.headers.get("ETag") if self.etag_cache is not None else None
            return payload, etag

        try:
            payload, etag = fetch(
                request,
                read,
                timeout=self.timeout,
                opener=self.opener,
                sleep=self.sleep,
            )
        except urllib.error.HTTPError as error:
            if error.code == 304 and cached:
                with self.lock:
                    self.requests += 1
                    self.not_modified += 1
                value = cached["body"]
            else:
                raise ProviderHttpError(url, error) from error
        else:
            with self.lock:
                self.requests += 1
            if limit is not None and len(payload) > limit:
                raise ProviderTooLarge(f"provider response exceeds {limit} bytes")
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ProviderDecodeError(str(error)) from error
            if etag and len(payload) < 2_000_000 and self.etag_cache is not None:
                with self.lock:
                    self.etag_cache[key] = {"etag": etag, "body": value}
        with self.lock:
            self.memo[key] = value
        return value
