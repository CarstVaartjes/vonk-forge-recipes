"""Time the sweep cannot observe the Controller is never load time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.policy import TimeoutPolicy
from spark_sweep.vonkctl import Reply, is_infrastructure


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _timeouts(seconds: float) -> TimeoutPolicy:
    return TimeoutPolicy(
        first_start_seconds=seconds, floor_seconds=seconds, cap_seconds=seconds
    )


def _outage(fleet, clock, code: str, detail: str, start: float, length: float):
    """Between start and start+length after now, every call but `update` fails."""
    begin = clock.now() + start
    real = fleet.runner

    def runner(argv, timeout):
        if begin <= clock.now() < begin + length and "update" not in argv:
            fleet.calls.append((None, tuple(argv[1:])))
            _, document = fleet._error(code, detail)
            return 2, json.dumps(document), ""
        return real(argv, timeout)

    fleet.runner = runner
    return fleet


def _make(tmp_path, gateway, recipe, limit):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [recipe],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=_timeouts(limit),
    )
    return sweep, fleet, clock


def _run(sweep, fleet):
    sweep.vk.runner = fleet.runner
    sweep.run()
    return sweep.state.recipes["vonk-forge/a"]


@pytest.mark.parametrize("code", ["http.502", "http.503", "http.504"])
def test_the_real_http_gateway_error_shape_is_infrastructure(code: str) -> None:
    reply = Reply(
        (),
        2,
        {
            "error_type": "control_api",
            "code": code,
            "detail": "control API request failed",
            "http_status": int(code[-3:]),
            "source": "remote_rejection",
        },
        "",
    )
    assert is_infrastructure(reply)


def test_http_502_and_503_are_infrastructure() -> None:
    for status in (502, 503):
        reply = Reply(
            (), 2, {"error_type": "control_api", "code": "x", "status": status}, ""
        )
        assert is_infrastructure(reply)


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("controller.transport_unavailable", "connection refused"),
        ("controller.unavailable", "HTTP 503"),
        (
            "controller.protocol_invalid",
            "control API response does not match the OpenAPI schema: 'install_partial' was unexpected",
        ),
    ],
)
def test_an_outage_during_a_load_does_not_fail_the_recipe(
    tmp_path: Path, gateway: Gateway, code: str, detail: str
) -> None:
    sweep, fleet, clock = _make(
        tmp_path, gateway, FakeRecipe("a", load_seconds=300), 600
    )
    _outage(fleet, clock, code, detail, start=100, length=3000)  # five times the limit
    entry = _run(sweep, fleet)
    assert entry["status"] == "passed", entry
    if code == "controller.protocol_invalid":
        assert any(
            "CLIENT MUST BE UPDATED" in e["message"] for e in sweep.state.data["events"]
        )
        assert any(c == ("update",) for _, c in fleet.calls)
        assert not any("--apply" in c for _, c in fleet.calls)


def test_a_genuine_not_serving_for_the_full_observed_limit_still_fails(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _make(
        tmp_path, gateway, FakeRecipe("a", load_seconds=10**6), 600
    )
    # an outage in the middle does not excuse the rest: the limit still runs while observed
    _outage(
        fleet, clock, "controller.transport_unavailable", "down", start=100, length=2000
    )
    entry = _run(sweep, fleet)
    assert (entry["status"], entry["failure_class"]) == ("failed", "timeout")
    assert entry["phase"] == "timeout"
