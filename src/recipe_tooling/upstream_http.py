"""The one place the tools open a provider URL: bounded retry, never a false finding.

A rate limit (429) or a "busy" answer (503) is transient: it is retried with exponential
backoff and jitter, honouring ``Retry-After`` (seconds or an HTTP date). If it persists the
item is "not checked this run" (``UpstreamBusy``); the next run asks again. The host is then
remembered as busy for the advised time, so the rest of the run does not queue behind it.
Other failures (outages, network errors, anything else) are our own faults and stay errors.
"""

from __future__ import annotations

import random
import ssl
import threading
import time
import urllib.request
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError

RETRY_CODES = frozenset({429, 500, 502, 503, 504})
BUSY_CODES = frozenset({429, 503})
ATTEMPTS = 4
MAX_WAIT_SECONDS = 30.0

_busy_until: dict[tuple[str, bool], float] = {}
_lock = threading.Lock()


class UpstreamBusy(RuntimeError):
    """The provider kept answering "try later"; this item was not checked this run."""


def forget_busy() -> None:
    """Drop the busy memory (a new run, or a test)."""
    with _lock:
        _busy_until.clear()


def _advised(error: HTTPError, now: float) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - now)
    except TypeError, ValueError:
        return None


def _backoff(attempt: int, jitter: Callable[[], float]) -> float:
    return min(MAX_WAIT_SECONDS, 2.0**attempt) * (0.5 + jitter())


def open_request(
    request: urllib.request.Request,
    timeout: float,
    opener: Callable[..., Any] | None,
    handlers: tuple[Any, ...],
) -> Any:
    if opener is not None:
        return opener(request, timeout=timeout)
    if handlers:
        return urllib.request.build_opener(*handlers).open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def open_once(
    request: urllib.request.Request | str,
    timeout: float,
    context: ssl.SSLContext | None = None,
) -> Any:
    """One raw open with no retry and no busy memory.

    For the callers that must not be retried: the App credential exchange (a security
    edge), a download that has its own jittered retry and mirror fallback, and a local
    fixture. Every `urlopen` of the repository is here.
    """
    if context is None:
        return urllib.request.urlopen(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def fetch[T](
    request: urllib.request.Request,
    read: Callable[[Any], T],
    *,
    timeout: float = 60,
    opener: Callable[..., Any] | None = None,
    handlers: tuple[Any, ...] = (),
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[], float] = random.random,
    clock: Callable[[], float] = time.time,
) -> T:
    """Open ``request`` and return ``read(response)``; retry transient answers in bounds.

    Raises ``UpstreamBusy`` when the host is rate limited past the bound, ``RuntimeError``
    when another transient failure outlasts it; any other HTTP error propagates unchanged.
    """
    # An authenticated read has its own quota: one anonymous 429 must not block it.
    host = (request.host, request.has_header("Authorization"))
    with _lock:
        waiting = _busy_until.get(host, 0.0) - clock()
    if waiting > 0:
        raise UpstreamBusy(f"{host[0]} is rate limited; not checked this run")
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            context = open_request(request, timeout, opener, handlers)
        except HTTPError as error:
            if error.code not in RETRY_CODES:
                raise
            advised = _advised(error, clock())
            if last or (advised is not None and advised > MAX_WAIT_SECONDS):
                if error.code in BUSY_CODES:
                    with _lock:
                        _busy_until[host] = clock() + (
                            advised if advised is not None else MAX_WAIT_SECONDS
                        )
                    raise UpstreamBusy(
                        f"provider returned HTTP {error.code}"
                    ) from error
                raise RuntimeError(f"provider returned HTTP {error.code}") from error
            wait = advised if advised is not None else _backoff(attempt, jitter)
        except (URLError, TimeoutError) as error:
            if last:
                raise RuntimeError(f"provider request failed: {error}") from error
            wait = _backoff(attempt, jitter)
        else:
            # Only opening is retried: a failure while reading the body is the caller's.
            with context as response:
                return read(response)
        (sleep or time.sleep)(wait)
    raise AssertionError("unreachable")
