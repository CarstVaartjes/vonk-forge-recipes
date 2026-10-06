"""Every recorded failure carries its evidence; the sweep survives its tool vanishing."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep
from test_sweep_vonkctl_smoke import _recipe

from spark_sweep import policy
from spark_sweep.definitions import Definitions
from spark_sweep.policy import classify, redact
from spark_sweep.smoke import (
    AssertionFailed,
    HttpConfig,
    _assertion_failure,
    _transport_failure,
    smoke_service,
)
from spark_sweep.vonkctl import Vonkctl


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


class _Bad:
    def __init__(self, status: int, body: bytes) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                return

            def _go(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(outer.status)
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                self.wfile.write(outer.body)

            do_GET = do_POST = _go

        self.status, self.body = status, body
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/x"


def test_http_400_keeps_case_status_and_a_redacted_bounded_body(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("s3cr3t-key-value\n")
    body = (
        b'{"error":"bad request, Authorization: Bearer abc.def-123 key s3cr3t-key-value "}'
        + b" " * 5000
    )
    server = _Bad(400, body)
    try:
        result = smoke_service(
            _recipe(),
            Definitions.load(),
            server.base,
            "x",
            HttpConfig(key_file=key),
        )
    finally:
        server.server.shutdown()
    failure = result.failure
    assert failure is not None
    assert failure.evidence["case"] == "stream"
    assert failure.evidence["http_status"] == 400
    text = failure.evidence["body"]
    assert "bad request" in text and len(text) <= 2048
    assert "abc.def-123" not in text and "s3cr3t-key-value" not in text
    assert "HTTP 400" in failure.describe() and "case=stream" in failure.describe()


def test_assertion_failure_keeps_the_offending_value() -> None:
    failure = _assertion_failure(
        AssertionFailed("path.equals failed", "wrong", "path.equals"), "A391"
    )
    assert failure.evidence["case"] == "A391" and failure.evidence["value"] == "wrong"


def test_redaction() -> None:
    assert "tok" not in redact(
        '{"api_key": "tok123", "x": 1} Bearer tok456 sk-abcdefgh12'
    )
    assert "keep" in redact("keep", ["", "zzz"])


def test_every_classified_failure_has_evidence() -> None:
    failures = [
        classify("download", "", ""),
        classify("review", "review.blocked", "not allowed"),
        classify("start", "application.failed", "container exited"),
        classify("readiness", "endpoint", "x"),
        classify("smoke", "smoke.crash", "ValueError()"),
        classify("timeout"),
        classify("install", "copy.stalled", "no progress", "copy-stalled"),
        _transport_failure(OSError("refused"), "M0"),
        _transport_failure(
            urllib.error.HTTPError(
                "http://x", 503, "unavailable", Message(), io.BytesIO(b"busy")
            ),
            "M0",
        ),
        _assertion_failure(AssertionFailed("nope"), "T_REPORT"),
    ]
    for failure in failures:
        assert failure.evidence, failure
        assert {"phase", "code", "detail"} <= set(failure.evidence)


def test_failed_load_row_carries_operation_and_reason(
    tmp_path: Path, gateway: Gateway
) -> None:
    bad = FakeRecipe(
        "bad",
        ("m2",),
        fail_load="container exited with code 1",
        fail_phase="start",
        fail_code="recipe.runtime_exit",
    )
    sweep, _, _ = make_sweep(tmp_path, [bad], [FakeModel("m2")], gateway=gateway)
    sweep.run()
    row = sweep.state.recipes["vonk-forge/bad"]
    proof = policy.evidence_of(row)
    assert row["status"] == "failed"
    assert proof["operation_id"] and "fleet evidence" in proof["evidence_command"]
    assert "container exited" in json.dumps(proof)
    status = (tmp_path / "status.md").read_text()
    assert "evidence:" in status and "operation_id=" in status


def test_a_failed_smoke_row_and_status_carry_the_evidence(
    tmp_path: Path, gateway: Gateway
) -> None:
    r = FakeRecipe("r")
    sweep, _, _ = make_sweep(tmp_path, [r], [FakeModel("m1")], gateway=gateway)
    gateway.broken.add(r.alias or "")
    sweep.run()
    row = sweep.state.recipes["vonk-forge/r"]
    if row["status"] == "failed":
        assert policy.evidence_of(row).get("case")


# -- the tool binary vanishing -----------------------------------------------------------------


def test_a_missing_vonkctl_is_infrastructure_not_a_crash() -> None:
    def missing(argv, timeout):
        raise FileNotFoundError(2, "No such file or directory", "vonkctl")

    reply = Vonkctl("vonkctl", runner=missing).run("fleet", "show")
    assert reply.document is None and "FileNotFoundError" in reply.error_text
    from spark_sweep.vonkctl import is_infrastructure

    assert is_infrastructure(reply)


def test_the_sweep_waits_out_a_missing_binary_at_start_and_mid_run(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("good")], [FakeModel("m1")], gateway=gateway
    )
    real = fleet.runner
    state = {"calls": 0}

    def flaky(argv, timeout):
        state["calls"] += 1
        # gone for the first calls (startup), and again around call 60 (mid-run)
        if state["calls"] <= 6 or 60 <= state["calls"] < 80:
            raise FileNotFoundError(2, "No such file or directory: 'vonkctl'")
        return real(argv, timeout)

    sweep.vk.runner = flaky
    assert sweep.run() == 0
    assert sweep.state.recipes["vonk-forge/good"]["status"] == "passed"
    assert state["calls"] > 80


def test_an_unexpected_exception_in_a_pass_is_an_infra_event_and_the_loop_continues(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("good")], [FakeModel("m1")], gateway=gateway
    )
    real = sweep._tick
    boom = {"n": 0}

    def tick() -> None:
        boom["n"] += 1
        if boom["n"] <= 2:
            raise KeyError("surprise")
        real()

    sweep._tick = tick  # type: ignore[method-assign]
    assert sweep.run() == 0
    assert sweep.state.recipes["vonk-forge/good"]["status"] == "passed"
    assert any("unexpected error" in str(e) for e in sweep.state.data.get("events", []))


def test_an_unexpected_exception_outside_the_pass_does_not_end_the_loop(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("good")], [FakeModel("m1")], gateway=gateway
    )
    real = sweep.done
    boom = {"n": 0}

    def done() -> bool:
        boom["n"] += 1
        if boom["n"] <= 2:
            raise RuntimeError("surprise")
        return real()

    sweep.done = done  # type: ignore[method-assign]
    assert sweep.run() == 0
    assert sweep.state.recipes["vonk-forge/good"]["status"] == "passed"


def test_an_interrupt_still_ends_the_loop(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("good")], [FakeModel("m1")], gateway=gateway
    )

    def tick() -> None:
        raise KeyboardInterrupt

    sweep._tick = tick  # type: ignore[method-assign]
    assert sweep.run() == 130
