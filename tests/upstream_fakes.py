"""Fake the one place provider URLs are opened (`recipe_tooling.upstream_http`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch


def patched_open(opens: Callable[..., Any]):
    """Patch `build_opener` so every provider open calls ``opens(request, timeout=...)``."""

    class Opener:
        def open(self, request: Any, timeout: float | None = None) -> Any:
            return opens(request, timeout=timeout)

    return patch("urllib.request.build_opener", return_value=Opener())
