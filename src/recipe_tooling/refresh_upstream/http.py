"""The HTTP client of the upstream refresh."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from recipe_tooling.kit_pins import upstream as kit_providers
from recipe_tooling.upstream_clients import (
    ProviderDecodeError,
    ProviderHttpError,
    ProviderJson,
)

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
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path and cache_path.is_file():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.cache = {}
        self.client = ProviderJson(
            user_agent="vonk-forge-refresh-upstream/1",
            timeout=30,
            tokens={"github": github_token, "huggingface": hf_token},
            etag_cache=self.cache,
        )
        self.kit_source: Any = None  # kit_upstream: one per run

    @property
    def requests(self) -> int:
        return self.client.requests

    @property
    def not_modified(self) -> int:
        return self.client.not_modified

    def save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.client.lock:
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def json(self, url: str, provider: str, *, anonymous: bool = False) -> Any:
        try:
            return self.client.get(url, provider, anonymous=anonymous)
        except ProviderHttpError as error:
            if anonymous and error.code in {401, 403}:
                raise AccessRestricted(
                    f"anonymous provider request returned HTTP {error.code}"
                ) from error.error
            if error.code == 404:
                raise NotFound(url, _error_message(error.error)) from error.error
            raise RuntimeError(f"HTTP {error.code} for {url}") from error.error
        except ProviderDecodeError as error:
            raise RuntimeError(f"request failed for {url}: {error}") from error

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
