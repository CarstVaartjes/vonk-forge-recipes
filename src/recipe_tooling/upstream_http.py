"""The one place the tools open a provider URL: bounded retry, never a false finding.

A rate limit (429) or a "busy" answer (503) is transient: it is retried with exponential
backoff and jitter, honouring ``Retry-After`` (seconds or an HTTP date). If it persists the
item is "not checked this run" (``UpstreamBusy``); the next run asks again. The host is then
remembered as busy for the advised time, so the rest of the run does not queue behind it.
A dropped or timed-out connection is retried the same way. Other failures (outages, definitive
network answers such as a certificate error, anything else) are our own faults and stay errors.
"""

from __future__ import annotations

import http.client
import random
import ssl
import threading
import time
import urllib.request
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

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
    except (TypeError, ValueError):
        return None


def _transient(error: BaseException) -> bool:
    """A dropped or timed-out connection (also inside a ``URLError``); not a definitive answer.

    A TLS certificate failure is an answer about the peer, never retried.
    """
    if isinstance(error, URLError) and not isinstance(error, HTTPError):
        return (
            _transient(error.reason)
            if isinstance(error.reason, BaseException)
            else True
        )
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    return isinstance(
        error,
        (
            http.client.RemoteDisconnected,
            ConnectionError,
            TimeoutError,
            ssl.SSLEOFError,
        ),
    )


def _backoff(attempt: int, jitter: Callable[[], float]) -> float:
    return min(MAX_WAIT_SECONDS, 2.0**attempt) * (0.5 + jitter())


class _SameOriginCredentials(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but send ``Authorization`` only to the origin it was made for.

    urllib copies every header to the redirect target. A provider that redirects to a
    presigned storage URL (Docker Hub blobs) answers 400 when that URL also carries a
    Bearer token; curl, requests and Go all drop the credential across origins.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        origin = urlsplit(req.full_url)
        target = urlsplit(newurl)
        if (origin.scheme, origin.netloc) != (target.scheme, target.netloc):
            for store in (redirected.headers, redirected.unredirected_hdrs):
                for name in [key for key in store if key.lower() == "authorization"]:
                    del store[name]
        return redirected


def open_request(
    request: urllib.request.Request,
    timeout: float,
    opener: Callable[..., Any] | None,
) -> Any:
    if opener is not None:
        return opener(request, timeout=timeout)
    return urllib.request.build_opener(_SameOriginCredentials).open(
        request, timeout=timeout
    )


def open_once(request: urllib.request.Request | str, timeout: float) -> Any:
    """One raw open with no retry and no busy memory.

    For the callers that must not be retried: the App credential exchange (a security
    edge), a download that has its own jittered retry and mirror fallback, and a local
    fixture. Every provider open of the repository is here.
    """
    return urllib.request.build_opener(_SameOriginCredentials).open(
        request, timeout=timeout
    )


def fetch[T](
    request: urllib.request.Request,
    read: Callable[[Any], T],
    *,
    timeout: float = 60,
    opener: Callable[..., Any] | None = None,
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
            context = open_request(request, timeout, opener)
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
        except (URLError, OSError, http.client.HTTPException) as error:
            if not _transient(error):
                raise
            if last:
                raise RuntimeError(f"provider request failed: {error}") from error
            wait = _backoff(attempt, jitter)
        else:
            # Only opening is retried: a failure while reading the body is the caller's.
            with context as response:
                return read(response)
        (sleep or time.sleep)(wait)
    raise AssertionError("unreachable")
