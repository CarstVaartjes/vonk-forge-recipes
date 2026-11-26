import http.client
import socket
import ssl
import urllib.request
from urllib.error import URLError

import pytest

from recipe_tooling.upstream_http import ATTEMPTS, fetch


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(errors):
    calls = []
    waits = []

    def opener(request, timeout):
        calls.append(request)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return _Response()

    request = urllib.request.Request("https://ghcr.io/v2/x")
    return (
        calls,
        waits,
        lambda: fetch(
            request,
            lambda response: "ok",
            opener=opener,
            sleep=waits.append,
            jitter=lambda: 0.0,
        ),
    )


def test_dropped_connection_is_retried_then_succeeds():
    calls, waits, run = _run([http.client.RemoteDisconnected("closed")])
    assert run() == "ok"
    assert len(calls) == 2
    assert len(waits) == 1


def test_wrapped_reset_is_retried():
    calls, _, run = _run([URLError(ConnectionResetError("reset"))])
    assert run() == "ok"
    assert len(calls) == 2


def test_always_dropping_raises_after_the_bound():
    calls, waits, run = _run([http.client.RemoteDisconnected("closed")] * ATTEMPTS)
    with pytest.raises(RuntimeError, match="provider request failed"):
        run()
    assert len(calls) == ATTEMPTS
    assert len(waits) == ATTEMPTS - 1


def test_certificate_error_is_definitive():
    error = URLError(ssl.SSLCertVerificationError("bad cert"))
    calls, waits, run = _run([error])
    with pytest.raises(RuntimeError, match="provider request failed") as raised:
        run()
    assert raised.value.__cause__ is error
    assert len(calls) == 1
    assert not waits


def test_temporary_dns_failure_is_retried():
    calls, _, run = _run([URLError(socket.gaierror(socket.EAI_AGAIN, "try again"))])
    assert run() == "ok"
    assert len(calls) == 2
