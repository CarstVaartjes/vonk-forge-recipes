"""One bounded retry for the upstream scanners' provider requests.

A rate limit or an outage answered by the provider is transient: it is retried with
exponential backoff and jitter, honouring ``Retry-After``. When the bound is spent the
item is "not checked this run" (``UpstreamBusy``), never a finding: the next run asks again.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError

TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})
ATTEMPTS = 4
MAX_WAIT_SECONDS = 30.0


class UpstreamBusy(RuntimeError):
    """The provider kept answering "try later"; this item was not checked this run."""


def _wait(error: HTTPError, attempt: int, jitter: Callable[[], float]) -> float | None:
    """Seconds to wait before the next attempt; ``None`` when waiting is not worth it."""
    try:
        advised = float(error.headers.get("Retry-After") or "")
    except ValueError:
        advised = None
    if advised is not None and advised >= 0:
        return advised if advised <= MAX_WAIT_SECONDS else None
    return min(MAX_WAIT_SECONDS, 2.0**attempt) * (0.5 + jitter())


def fetch[T](
    request: Callable[[], T],
    *,
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Run ``request``; retry transient failures within the bound, else ``UpstreamBusy``.

    Any other HTTP error propagates to the caller unchanged.
    """
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            return request()
        except HTTPError as error:
            if error.code not in TRANSIENT_CODES:
                raise
            wait = None if last else _wait(error, attempt, jitter)
            if wait is None:
                raise UpstreamBusy(f"provider returned HTTP {error.code}") from error
        except (URLError, TimeoutError) as error:
            if last:
                raise UpstreamBusy(f"provider request failed: {error}") from error
            wait = min(MAX_WAIT_SECONDS, 2.0**attempt) * (0.5 + jitter())
        (sleep or time.sleep)(wait)
    raise AssertionError("unreachable")
