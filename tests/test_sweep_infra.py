"""Infrastructure errors never fail a recipe; one sweep per state directory; interrupts really end it."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]

from spark_sweep.state import StateLock, StateLocked
from spark_sweep.vonkctl import (
    Reply,
    Vonkctl,
    VonkctlTimeout,
    is_infrastructure,
    subprocess_runner,
)


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _error(code: str, detail: str = "x", error_type: str = "control_api") -> Reply:
    return Reply(
        (),
        2,
        {"error": detail, "error_type": error_type, "code": code, "detail": detail},
        "",
    )


# -- what counts as infrastructure ---------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "reviewing", "expected"),
    [
        (
            _error(
                "controller.protocol_invalid",
                "control API response does not match the OpenAPI schema",
            ),
            False,
            True,
        ),
        (_error("controller.transport_timeout"), False, True),
        (_error("controller.transport_unavailable"), False, True),
        (_error("controller.unavailable", "control API unavailable"), False, True),
        (_error("controller.invalid_request", "bad request"), True, True),
        (_error("controller.invalid_request", "bad request"), False, True),
        (_error("", "unknown command", error_type="arguments"), False, True),
        # Controller error prose cannot attribute infrastructure failure to a recipe.
        (
            _error(
                "controller.unavailable",
                "dockerfile.heredoc_forbidden: Dockerfile heredocs are not accepted",
            ),
            False,
            True,
        ),
        (_error("recipe.not_found", "no such recipe"), True, True),
        (Reply((), 124, None, "timed out"), False, True),
    ],
)
def test_client_protocol_and_transport_errors_are_infrastructure(
    reply: Reply, reviewing: bool, expected: bool
) -> None:
    assert is_infrastructure(reply, reviewing=reviewing) is expected


# -- they never fail a recipe ----------------------------------------------------------------


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


@pytest.mark.parametrize(
    ("prefix", "code", "detail"),
    [
        (("profile", "add"), "controller.invalid_request", "request does not match"),
        (
            ("profile", "add"),
            "controller.protocol_invalid",
            "control API response does not match the OpenAPI schema",
        ),
        (
            ("profile", "load", "--review"),
            "controller.protocol_invalid",
            "control API response does not match the OpenAPI schema",
        ),
        (("profile", "load", "--review"), "controller.invalid_request", "invalid"),
        (
            ("profile", "load", "--yes"),
            "controller.transport_timeout",
            "request failed [retry]",
        ),
        (
            ("recipe", "download"),
            "controller.protocol_invalid",
            "does not match the OpenAPI schema",
        ),
        (
            ("recipe", "download"),
            "controller.transport_unavailable",
            "connection refused",
        ),
    ],
)
@pytest.mark.usefixtures("bounded_clock")
def test_an_infrastructure_error_pauses_and_retries_but_fails_no_recipe(
    tmp_path: Path, gateway: Gateway, prefix: tuple[str, ...], code: str, detail: str
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.faults = [(prefix, code, detail)] * 2
    seen: list[dict] = []
    clock.hooks.append(
        lambda _t: seen.append(json.loads(json.dumps(sweep.state.data["infra"])))
    )
    assert sweep.run() == 0
    assert _entry(sweep, "a")["status"] == "passed"  # not marked FAILED
    assert _entry(sweep, "a")["attempts"] == 1
    assert any(item for item in seen), "the problem was surfaced while it lasted"
    assert sweep.state.data["infra"] == {}  # and cleared once the Controller answered


def test_the_status_page_shows_infrastructure_problems_with_the_update_hint(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a")],
        [FakeModel("m1")],
        gateway=gateway,
        status_seconds=0,
    )
    fleet.faults = [
        (
            ("profile", "load", "--review"),
            "controller.protocol_invalid",
            "control API response does not match the OpenAPI schema",
        )
    ]
    pages: list[str] = []
    clock.hooks.append(
        lambda _t: (
            pages.append((tmp_path / "status.md").read_text())
            if (tmp_path / "status.md").exists()
            else None
        )
    )
    sweep.run()
    page = next(p for p in pages if "Infrastructure problems" in p)
    assert "not recipe failures" in page and "vonkctl update --apply" in page
    statuses = [json.loads(line) for line in [(tmp_path / "status.json").read_text()]]
    assert "infrastructure" in statuses[0]


def test_backoff_grows_and_is_capped(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    waits = []
    for _ in range(8):
        sweep.note_infra("review", "controller.protocol_invalid")
        waits.append(sweep.state.data["infra"]["review"]["until"] - clock.now())
    assert waits[:3] == [30, 60, 120] and waits[-1] == 600


def test_a_recipe_specific_refusal_still_fails_the_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path,
        [FakeRecipe("a", fail_download="policy")],
        [FakeModel("m1")],
        gateway=gateway,
    )
    sweep.run()
    assert _entry(sweep, "a")["failure_class"] == "build-policy"


# -- the client version ----------------------------------------------------------------------


def test_a_skewed_client_updates_and_admits_work(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.accepted_version = "1.1"
    assert sweep.run() == 0
    assert sweep.state.data["client"]["version"] == "1.1"
    assert _entry(sweep, "a")["status"] == "passed"


def test_version_skew_can_be_accepted_loudly(tmp_path: Path, gateway: Gateway) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [FakeRecipe("a")],
        [FakeModel("m1")],
        gateway=gateway,
        allow_version_skew=True,
    )
    fleet.accepted_version = "1.1"
    assert sweep.run() == 0
    assert any(
        "WARNING: vonkctl 1.0 differs" in e["message"]
        for e in sweep.state.data["events"]
    )
    assert sweep.state.data["client"]["version"] == "1.0"


def test_an_unanswered_version_check_warns_but_does_not_block(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.faults = [(("update",), "controller.transport_timeout", "no network")]
    assert sweep.run() == 0
    assert any(
        "could not compare vonkctl" in e["message"] for e in sweep.state.data["events"]
    )


def test_the_cli_updates_a_skewed_client(
    tmp_path: Path, gateway: Gateway, capsys
) -> None:
    from sweep_fakes import make_definitions

    from spark_sweep.cli import main

    recipes = [FakeRecipe("a")]
    _, fleet, clock = make_sweep(
        tmp_path / "s", recipes, [FakeModel("m1")], gateway=gateway
    )
    fleet.accepted_version = "1.1"
    code = main(
        ["run", "--yes", "--state-dir", str(tmp_path / "s")],
        vonkctl_factory=lambda exe: Vonkctl(exe, runner=fleet.runner),
        clock=clock,
        definitions=make_definitions(recipes),
    )
    assert code == 0 and fleet.client_build["version"] == "1.1"


# -- one sweep per state directory -------------------------------------------------------------


def test_a_second_sweep_on_the_same_state_directory_refuses_to_start(
    tmp_path: Path, gateway: Gateway
) -> None:
    first, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    second, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], fleet=fleet
    )
    outcome: list[str] = []

    def start_second(_now: float) -> None:
        if not outcome:
            try:
                second.run()
            except StateLocked as error:
                outcome.append(str(error))

    clock.hooks.append(start_second)
    assert first.run() == 0
    assert (
        f"pid {os.getpid()}" in outcome[0] and "another sweep is running" in outcome[0]
    )
    # The refused sweep touched nothing, and once the first one ended the lock is free again.
    third, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], fleet=fleet
    )
    assert third.run() == 0


def test_the_lock_is_released_when_a_run_crashes_and_names_its_holder(
    tmp_path: Path,
) -> None:
    lock = StateLock(tmp_path)
    lock.acquire()
    assert (
        "pid " in (tmp_path / "run.lock").read_text()
        and "on " in (tmp_path / "run.lock").read_text()
    )
    with pytest.raises(StateLocked):
        StateLock(tmp_path).acquire()
    lock.release()
    other = StateLock(tmp_path)
    other.acquire()  # a crash closes the file and frees the flock: no stale lock to clean up
    other.release()


def test_the_lock_survives_in_another_process(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, time; sys.path.insert(0, sys.argv[1]); "
                "from pathlib import Path; from spark_sweep.state import StateLock; "
                "l = StateLock(Path(sys.argv[2])); l.acquire(); "
                "print('held', flush=True); time.sleep(30)"
            ),
            str(Path(__file__).resolve().parents[1]),
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(StateLocked, match=f"pid {holder.pid}"):
            StateLock(tmp_path).acquire()
    finally:
        holder.kill()
        holder.wait()
    again = StateLock(tmp_path)
    again.acquire()
    again.release()


# -- interrupts ----------------------------------------------------------------------------------


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "slow-vonkctl"
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_sigint_kills_a_blocked_vonkctl_and_returns_promptly(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    command = _script(
        tmp_path,
        f"import os, time\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n",
    )
    interrupted_at: list[float] = []

    def interrupt_started_child() -> None:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        interrupted_at.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_started_child)
    interrupter.start()
    with pytest.raises(KeyboardInterrupt):
        subprocess_runner([command], 60)
    interrupter.join(timeout=1)
    assert interrupted_at and time.monotonic() - interrupted_at[0] < 5
    time.sleep(0.2)
    assert not _alive(
        int(pidfile.read_text())
    )  # the child did not outlive the interrupt
    code, stdout, _ = subprocess_runner([sys.executable, "-c", "print('fresh')"], 5)
    assert code == 0 and stdout.strip() == "fresh"


def test_a_vonkctl_that_ignores_sigint_and_hangs_is_killed_at_the_timeout(
    tmp_path: Path,
) -> None:
    pidfile = tmp_path / "pid"
    command = _script(
        tmp_path,
        f"import os, signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n",
    )
    started = time.monotonic()
    with pytest.raises(VonkctlTimeout):
        subprocess_runner([command], 1)
    assert time.monotonic() - started < 5
    time.sleep(0.2)
    assert not _alive(int(pidfile.read_text()))


def test_a_sweep_started_with_sigint_ignored_still_stops_on_sigint(
    tmp_path: Path, gateway: Gateway
) -> None:
    # A process started in the background of a non-interactive shell inherits SIGINT as ignored.
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", load_seconds=900)],
        [FakeModel("m1")],
        gateway=gateway,
    )
    original = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        sent: list[bool] = []

        def interrupt(now: float) -> None:
            if now - 1_000_000.0 >= 200 and not sent:
                sent.append(True)
                os.kill(os.getpid(), signal.SIGINT)

        clock.hooks.append(interrupt)
        assert sweep.run() == 130
        assert (
            signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        )  # the previous handler is restored
    finally:
        signal.signal(signal.SIGINT, original)
    assert any(c[:2] == ("profile", "cancel") for _, c in fleet.calls)


def test_sigterm_stops_a_sweep_like_sigint(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", load_seconds=900)],
        [FakeModel("m1")],
        gateway=gateway,
    )
    sent: list[bool] = []

    def terminate(now: float) -> None:
        if now - 1_000_000.0 >= 200 and not sent:
            sent.append(True)
            os.kill(os.getpid(), signal.SIGTERM)

    clock.hooks.append(terminate)
    assert sweep.run() == 130


def test_a_hung_smoke_request_does_not_keep_an_interrupted_sweep_alive() -> None:
    from spark_sweep.run import DaemonExecutor

    release = threading.Event()
    future = DaemonExecutor().submit(release.wait, 30)
    assert not future.done()
    thread = next(t for t in threading.enumerate() if t.name == "smoke")
    assert thread.daemon  # the interpreter will not wait for it at exit
    release.set()
    assert future.result(timeout=5) is True


@pytest.mark.parametrize("accepted_then_lost", [False, True])
def test_download_reconnect_preserves_the_original_request(
    tmp_path: Path, accepted_then_lost: bool
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("reconnect")], [FakeModel("m1")]
    )
    sweep.refresh_catalog()
    recipe = sweep.recipes["vonk-forge/reconnect"]
    original = sweep.vk.runner
    lost = False

    def run(argv, timeout):
        nonlocal lost
        if "download" in argv and not lost:
            lost = True
            if accepted_then_lost:
                original(argv, timeout)
            return (
                2,
                json.dumps(
                    {"error_type": "control_api", "code": "controller.protocol_invalid"}
                ),
                "",
            )
        return original(argv, timeout)

    sweep.vk.runner = run
    sweep.prefetcher._request(recipe, "model")
    before = dict(sweep.state.downloads[recipe.key])
    assert before["state"] == "observing" and not before.get("operation_id")
    clock.sleep(60)
    sweep.prefetcher._poll()
    after = sweep.state.downloads[recipe.key]
    assert after["operation_id"]
    assert after["request_key"] == before["request_key"]
    assert after["started_at"] == before["started_at"]
    assert after["attempt"] == before["attempt"] == 1
    assert after["accepted_intent"] == before["requested_intent"]
    calls = [call for _, call in fleet.calls if call[:2] == ("recipe", "download")]
    assert len(calls) == 1
    assert calls[0][-1] == before["request_key"]
    assert not sweep.state.data["download_history"]


def test_accepted_download_absence_keeps_unknown_parent_without_replay(
    tmp_path: Path,
) -> None:
    sweep, fleet, _clock = make_sweep(
        tmp_path, [FakeRecipe("accepted")], [FakeModel("m1")]
    )
    sweep.refresh_catalog()
    recipe = sweep.recipes["vonk-forge/accepted"]
    sweep.prefetcher._request(recipe, "model")
    before = dict(sweep.state.downloads[recipe.key])
    fleet.observations.append(
        (
            ("recipe", "progress"),
            2,
            {"error_type": "control_api", "code": "controller.not_found"},
        )
    )
    sweep.prefetcher._poll()
    after = sweep.state.downloads[recipe.key]
    assert after["operation_id"] == before["operation_id"]
    assert after["request_key"] == before["request_key"]
    assert after["attempt"] == before["attempt"]
    assert (
        len([call for _, call in fleet.calls if call[:2] == ("recipe", "download")])
        == 1
    )
    assert not sweep.state.data["download_history"]
