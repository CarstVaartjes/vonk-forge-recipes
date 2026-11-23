"""The MiniMax H3 vLLM-Omni job wrapper, both ranks, with fakes at the process boundary only.

The fake agent port is a real loopback listener, the fake server a real child process
speaking HTTP, the fake agent a real child process writing a log. No GPU, no vLLM.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import signal
import socket
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

import pytest
from repo_root import ROOT

ADAPTER = ROOT / "adapters/video/minimax-h3-vllm-omni-dual"
_SPEC = importlib.util.spec_from_file_location(
    "vllm_omni_job", ADAPTER / "vllm_omni_job.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
job = importlib.util.module_from_spec(_SPEC)
sys.modules["vllm_omni_job"] = job
_SPEC.loader.exec_module(job)

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64

FAKE_SERVER = textwrap.dedent(
    """
    import http.server, os, sys, time
    port = int(sys.argv[sys.argv.index("--port") + 1])
    mode = os.environ["FAKE_MODE"]
    marker = os.environ.get("FAKE_MARKER")
    if marker:
        open(marker, "w").write("started")
    if mode == "dies-starting":
        sys.exit(3)
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404); self.end_headers()
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if marker:
                open(marker + ".post", "w").write("posted")
            if mode == "http500":
                self.send_response(500); self.end_headers(); self.wfile.write(b"engine exploded"); return
            if mode == "slow":
                time.sleep(60)
            body = b"not a video" if mode == "not-mp4" else open(os.environ["FAKE_VIDEO"], "rb").read()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)

FAKE_AGENT = textwrap.dedent(
    """
    import os, sys, time
    log_dir = sys.argv[1]
    os.makedirs(log_dir, exist_ok=True)
    if os.environ.get("FAKE_CHILD_PIDFILE"):
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        open(os.path.join(log_dir, "rank1.pid"), "w").write(str(child.pid))
        open(os.environ["FAKE_CHILD_PIDFILE"], "w").write(str(child.pid))
    with open(os.path.join(log_dir, "worker-rank1-1.log"), "ab") as log:
        log.write(b"rank1: ready\\n"); log.flush()
    if os.environ.get("FAKE_AGENT_EXIT"):
        sys.exit(int(os.environ["FAKE_AGENT_EXIT"]))
    time.sleep(120)
    """
)

FAST = job.Timing(
    agent_ready_s=2.0,
    probe_first_s=0.02,
    probe_cap_s=0.1,
    health_cap_s=10.0,
    health_poll_s=0.02,
    request_s=10.0,
    request_margin_s=1.0,
    render_poll_s=0.02,
    server_grace_s=2.0,
    agent_grace_s=2.0,
    worker_grace_s=1.0,
    tail_poll_s=0.02,
)


class FakeClock:
    """Time that only moves when a wait is asked for; records every wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []
        self.on_wait: Callable[[int], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def wait(self, cancel: threading.Event, seconds: float) -> bool:
        self.waits.append(seconds)
        self.now += seconds
        if self.on_wait is not None:
            self.on_wait(len(self.waits))
        return cancel.is_set()


def environment(rank: int, **overrides: str) -> dict[str, str]:
    env = {
        "VONK_RANK": str(rank),
        "VONK_WORLD_SIZE": "2",
        "VONK_LOCAL_ADDR": "127.0.0.1",
        "VONK_MASTER_ADDR": "127.0.0.1",
        "VONK_MASTER_PORT": "29500",
        "VONK_PEER_ADDRS": "127.0.0.1,127.0.0.1",
        "VONK_JOB_TIMEOUT_SECONDS": "40",
    }
    env.update(overrides)
    return env


@pytest.fixture(autouse=True)
def said(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every line the wrapper prints through ``say``, in order."""
    lines: list[str] = []
    real = job.say

    def record(message: str) -> None:
        lines.append(f"vllm-omni-job: {message}")
        real(message)

    monkeypatch.setattr(job, "say", record)
    return lines


@contextlib.contextmanager
def trigger(ready: Callable[[], bool], action: Callable[[], None]) -> Generator[None]:
    """Run ``action`` once ``ready()`` holds (events, never fixed delays); never after the block ended."""
    done = threading.Event()

    def watch() -> None:
        deadline = time.monotonic() + 60
        while not done.is_set() and time.monotonic() < deadline:
            if ready():
                action()
                return
            time.sleep(0.005)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=5)


def settings(task: str = "fl2va", **overrides: object) -> Any:
    values: dict[str, object] = {
        "task": task,
        "model": "/models/FL2VA",
        "duration": 5.0,
        "width": 1344,
        "height": 768,
        "steps": 20,
        "flow_shift": 12.0,
        "seed": 1101,
    }
    values.update(overrides)
    return job.Settings(**values)


def png(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return path


def stage(root: Path, **slots: list[str]) -> None:
    """Write the platform's input manifest: each named file belongs to its slot."""
    files = [
        {"slot": slot.replace("_", "-"), "name": name}
        for slot, names in slots.items()
        for name in names
    ]
    (root / "manifest.json").write_text(json.dumps({"files": files}))


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "prompt.txt").write_text("A quiet valley at dawn.\n", encoding="utf-8")
    stage(root, prompt=["prompt.txt"])
    return root


def fields(request: Any) -> dict[str, str]:
    return dict(request.fields)


# --- environment ----------------------------------------------------------------------------


def test_placement_maps_the_platform_environment() -> None:
    place = job.read_placement(
        environment(0, VONK_PEER_ADDRS="10.0.0.1,10.0.0.2", VONK_MASTER_ADDR="10.0.0.1")
    )
    assert (place.rank, place.worker, place.master_port, place.job_timeout_s) == (
        0,
        "10.0.0.2",
        29500,
        40,
    )
    head = job.child_environment(place, {"PATH": "/bin", "H3_TP2_SECRET": "inherited"})
    assert head["H3_TP2_HEAD_IP"] == "10.0.0.1"
    assert head["H3_TP2_WORKER_IP"] == "10.0.0.2"
    assert head["H3_TP2_WORKER_AGENT"] == f"10.0.0.2:{job.AGENT_PORT}"
    assert head["H3_TP2_ROLE"] == "head"
    assert head["MASTER_PORT"] == "29500"
    assert "H3_TP2_SECRET" not in head  # the creator's channel runs without HMAC
    assert head["VLLM_HOST_IP"] == "127.0.0.1"
    assert head["H3_TP2_WORLD_SIZE"] == "2"
    worker = job.child_environment(job.read_placement(environment(1)), {})
    assert "H3_TP2_ROLE" not in worker
    assert "MASTER_PORT" not in worker


@pytest.mark.parametrize(
    ("override", "named"),
    [
        ({"VONK_WORLD_SIZE": "3"}, "VONK_WORLD_SIZE"),
        ({"VONK_RANK": "2"}, "VONK_RANK"),
        ({"VONK_LOCAL_ADDR": ""}, "VONK_LOCAL_ADDR"),
        ({"VONK_MASTER_PORT": "70000"}, "VONK_MASTER_PORT"),
        ({"VONK_PEER_ADDRS": "10.0.0.1"}, "VONK_PEER_ADDRS"),
    ],
)
def test_the_wrapper_refuses_to_start_and_names_only_the_variable(
    override: dict[str, str],
    named: str,
    inputs: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = environment(0, **override)
    argv = [
        "--model=/models/FL2VA", "--task=fl2va", "--duration=5",
        "--width=1344", "--height=768", "--num-inference-steps=20", "--flow-shift=12", "--seed=1101",
        f"--output-dir={inputs}",
    ]  # fmt: skip
    assert job.main(argv, env) == job.EXIT_CONFIG
    err = capsys.readouterr().err
    assert named in err


@pytest.mark.parametrize(
    ("timeout", "interval"),
    [(None, 30.0), (1, 5.0), (40, 10.0), (3600, 60.0)],
)
def test_heartbeat_is_a_quarter_of_the_submitters_window(
    timeout: int | None, interval: float
) -> None:
    assert job.heartbeat_interval(timeout) == interval


# --- the request ----------------------------------------------------------------------------


def test_a_text_only_request_is_t2va_with_the_creators_defaults(inputs: Path) -> None:
    request = job.build_request(settings(), inputs)
    form = fields(request)
    assert (
        form["width"],
        form["height"],
        form["num_inference_steps"],
        form["fps"],
    ) == ("1344", "768", "20", "24")
    assert (form["flow_shift"], form["seed"], form["aspect_ratio"]) == (
        "12",
        "1101",
        "16:9",
    )
    assert form["prompt"] == "A quiet valley at dawn.\n"
    assert json.loads(form["extra_params"]) == {
        "task": "t2va",
        "duration": 5,
        "audio_flow_shift": 3.0,
    }
    assert request.images == ()


def test_keyframes_choose_the_task_and_the_frame_indices(inputs: Path) -> None:
    png(inputs / "a.png")
    png(inputs / "b.png")
    cases = [
        ({"first_image": ["a.png"]}, [0], "input_reference", 1),
        ({"last_image": ["b.png"]}, [-1], "input_reference", 1),
        (
            {"first_image": ["a.png"], "last_image": ["b.png"]},
            [0, -1],
            "input_references",
            2,
        ),
    ]
    for slots, indices, field_name, count in cases:
        stage(inputs, **slots)
        request = job.build_request(settings(duration=10.0, seed=7), inputs)
        extra = json.loads(fields(request)["extra_params"])
        assert extra == {
            "task": "fl2va",
            "duration": 10,
            "audio_flow_shift": 3.0,
            "frame_indices": indices,
        }
        assert (request.image_field, len(request.images), fields(request)["seed"]) == (
            field_name,
            count,
            "7",
        )


def test_multipart_carries_every_field_and_each_image_under_the_right_name(
    inputs: Path,
) -> None:
    png(inputs / "a.png")
    png(inputs / "b.png")
    stage(inputs, first_image=["a.png"], last_image=["b.png"])
    body, content_type = job.encode_multipart(job.build_request(settings(), inputs))
    boundary = content_type.split("boundary=")[1].encode()
    assert body.count(b'name="input_references"') == 2
    assert b'name="extra_params"' in body
    assert body.endswith(b"--" + boundary + b"--\r\n")
    assert b"Content-Type: image/png" in body


def test_ref2va_is_long_video_continuation_from_references(
    inputs: Path,
) -> None:
    png(inputs / "1-scene.png")
    png(inputs / "2-face.png")
    stage(inputs, references=["2-face.png", "1-scene.png"])  # name order wins
    cfg = settings("ref2va", model="/models/Ref2VA", duration=30.0)
    request = job.build_request(cfg, inputs)
    extra = json.loads(fields(request)["extra_params"])
    assert extra == {
        "task": "ref2va", "duration": 30, "audio_flow_shift": 3.0,
        "long_video": True, "long_video_mode": "continuation",
    }  # fmt: skip
    assert request.image_field == "input_references"
    assert [image.name for image in request.images] == ["1-scene.png", "2-face.png"]


# --- the state machine ----------------------------------------------------------------------


def test_the_machine_refuses_an_edge_the_table_does_not_have() -> None:
    machine = job.Machine(0, job.HEAD_EDGES, job.HeadState.MAPPED, FakeClock())
    with pytest.raises(RuntimeError, match="no edge MAPPED -> RENDERING"):
        machine.step(job.HeadState.RENDERING)
    machine.step(job.HeadState.AWAIT_AGENT)
    assert machine.state is job.HeadState.AWAIT_AGENT


def test_unreachable_agent_backs_off_then_fails_at_its_bound(
    inputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: list[str]
) -> None:
    clock = FakeClock()
    ctx = job.Context(
        job.read_placement(environment(0)), settings(), {}, tmp_path,
        clock=clock, timing=job.Timing(agent_ready_s=50.0), heartbeat_s=3600.0,
        request=job.build_request(settings(), inputs),
    )  # fmt: skip
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    monkeypatch.setattr(job, "AGENT_PORT", free.getsockname()[1])
    free.close()
    assert job.HeadJob(ctx).run() == job.EXIT_FAILED
    assert clock.waits[:6] == [1.0, 2.0, 4.0, 8.0, 15.0, 15.0]
    assert any("did not answer within 50 s" in line for line in said)


@pytest.fixture
def stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., Any]]:
    """A loopback agent port, a fake server child and a private TMPDIR; yields a context factory."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    monkeypatch.setattr(job, "AGENT_PORT", listener.getsockname()[1])
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER)
    video = tmp_path / "fake.mp4"
    video.write_bytes(MP4)
    monkeypatch.setattr(
        job,
        "serve_argv",
        lambda model, port: [sys.executable, str(server), "--port", str(port)],
    )
    out = tmp_path / "outputs"
    out.mkdir()

    def make(mode: str, inputs: Path, **kwargs: object) -> Any:
        env = {
            **os.environ,
            "FAKE_MODE": mode,
            "FAKE_VIDEO": str(video),
            "FAKE_MARKER": str(tmp_path / "started"),
        }
        return job.Context(
            job.read_placement(environment(0)), settings(), env, out,
            timing=FAST, request=job.build_request(settings(), inputs), heartbeat_s=0.05, **kwargs,
        )  # fmt: skip

    yield make
    listener.close()


def run_head(ctx: Any, said: list[str]) -> tuple[int, list[str]]:
    code = job.HeadJob(ctx).run()
    return code, list(said)


def test_a_render_ends_as_an_atomic_mp4_and_a_stopped_server(
    stack: Callable[..., Any],
    inputs: Path,
    tmp_path: Path,
    said: list[str],
) -> None:
    code, log = run_head(stack("ok", inputs), said)
    assert code == 0
    assert (tmp_path / "outputs" / "video.mp4").read_bytes() == MP4
    assert sorted(path.name for path in (tmp_path / "outputs").iterdir()) == [
        "video.mp4"
    ]
    assert list((tmp_path / "tmp").iterdir()) == []  # the wrapper's own scratch is gone
    steps = [line.split("rank 0 ")[1].split(" (")[0] for line in log if "->" in line]
    assert steps == [
        "MAPPED -> AWAIT_AGENT", "AWAIT_AGENT -> SERVING", "SERVING -> HEALTHY", "HEALTHY -> RENDERING",
        "RENDERING -> WRITTEN", "WRITTEN -> STOPPING", "STOPPING -> DONE",
    ]  # fmt: skip


def test_the_heartbeat_names_the_state_while_the_render_runs(
    stack: Callable[..., Any], inputs: Path, said: list[str]
) -> None:
    ctx = stack("slow", inputs)

    def beating() -> bool:
        return any(
            line.startswith("vllm-omni-job: rank 0 RENDERING ") and "->" not in line
            for line in said
        )

    with trigger(beating, ctx.cancel.set):
        code, log = run_head(ctx, said)
    assert code == job.EXIT_CANCELLED
    assert any(
        line.startswith("vllm-omni-job: rank 0 RENDERING ") and line.endswith("s")
        for line in log
    )


def test_a_server_that_dies_while_starting_fails_the_job(
    stack: Callable[..., Any],
    inputs: Path,
    tmp_path: Path,
    said: list[str],
) -> None:
    code, log = run_head(stack("dies-starting", inputs), said)
    assert code == job.EXIT_FAILED
    assert not (tmp_path / "outputs" / "video.mp4").exists()
    assert any("exited with status 3 before it was healthy" in line for line in log)
    assert log[-2].endswith("STOPPING -> DONE")


@pytest.mark.parametrize(
    ("mode", "reason"),
    [("http500", "HTTP 500: engine exploded"), ("not-mp4", "not an MP4")],
)
def test_a_failed_render_fails_the_job_and_leaves_no_video(
    stack: Callable[..., Any], inputs: Path, tmp_path: Path, said: list[str],
    mode: str, reason: str,
) -> None:  # fmt: skip
    code, log = run_head(stack(mode, inputs), said)
    assert code == job.EXIT_FAILED
    assert list((tmp_path / "outputs").iterdir()) == []
    assert any(reason in line for line in log)


def test_an_agent_that_never_answers_fails_before_the_model_is_started(
    stack: Callable[..., Any], inputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    said: list[str],
) -> None:  # fmt: skip
    ctx = stack("ok", inputs)
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    monkeypatch.setattr(job, "AGENT_PORT", free.getsockname()[1])
    free.close()
    ctx.timing = job.Timing(**{**FAST.__dict__, "agent_ready_s": 0.3})
    code, log = run_head(ctx, said)
    assert code == job.EXIT_FAILED
    assert not (tmp_path / "started").exists()
    assert any("did not answer within" in line for line in log)


def test_an_agent_that_comes_up_late_is_waited_for(
    stack: Callable[..., Any], inputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    said: list[str],
) -> None:  # fmt: skip
    late = socket.socket()
    late.bind(("127.0.0.1", 0))
    monkeypatch.setattr(job, "AGENT_PORT", late.getsockname()[1])
    threading.Timer(0.3, late.listen, args=(4,)).start()
    code, _ = run_head(stack("ok", inputs), said)
    late.close()
    assert code == 0
    assert (tmp_path / "outputs" / "video.mp4").exists()


def test_sigterm_cancels_a_running_render_and_stops_the_server(
    stack: Callable[..., Any], inputs: Path, tmp_path: Path, said: list[str]
) -> None:
    ctx = stack("slow", inputs)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    job.install_term_handlers(ctx.cancel)
    posted = tmp_path / "started.post"
    try:
        with trigger(posted.exists, lambda: os.kill(os.getpid(), signal.SIGTERM)):
            code, log = run_head(ctx, said)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert code == job.EXIT_CANCELLED
    assert not (tmp_path / "outputs" / "video.mp4").exists()
    assert any("RENDERING -> STOPPING" in line for line in log)


def test_stop_group_escalates_to_sigkill_when_sigterm_is_ignored(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    code = f"import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
    proc = job.spawn([sys.executable, "-c", code], os.environ, tmp_path)
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    job.stop_group(proc, 0.3, job.RealClock())
    assert proc.poll() == -signal.SIGKILL


# --- rank 1 ---------------------------------------------------------------------------------


@pytest.fixture
def worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[Any, io.BytesIO]]:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    agent = tmp_path / "fake_agent.py"
    agent.write_text(FAKE_AGENT)
    monkeypatch.setattr(
        job,
        "agent_argv",
        lambda place, log_dir: [sys.executable, str(agent), str(log_dir)],
    )

    def make(**env: str) -> tuple[Any, io.BytesIO]:
        sink = io.BytesIO()
        ctx = job.Context(
            job.read_placement(environment(1)), settings(), {**os.environ, **env}, tmp_path,
            timing=FAST, heartbeat_s=0.05, log_out=sink,
        )  # fmt: skip
        return ctx, sink

    return make


def test_rank_one_streams_the_agent_log_until_it_is_stopped(
    worker: Callable[..., tuple[Any, io.BytesIO]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx, sink = worker()
    with trigger(lambda: sink.getvalue() == b"rank1: ready\n", ctx.cancel.set):
        code = job.WorkerJob(ctx).run()
    out = capsys.readouterr().out
    assert code == job.EXIT_CANCELLED  # asked to stop: not a failure
    assert sink.getvalue() == b"rank1: ready\n"
    assert "rank 1 AGENT -> STOPPING" in out


def test_rank_one_fails_when_the_agent_exits_and_still_shows_its_last_lines(
    worker: Callable[..., tuple[Any, io.BytesIO]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx, sink = worker(FAKE_AGENT_EXIT="1")
    code = job.WorkerJob(ctx).run()
    out = capsys.readouterr().out
    assert code == job.EXIT_FAILED
    assert sink.getvalue() == b"rank1: ready\n"
    assert "the agent exited with status 1" in out


def test_rank_one_ends_the_group_the_agent_recorded(
    worker: Callable[..., tuple[Any, io.BytesIO]], tmp_path: Path
) -> None:
    pidfile = tmp_path / "rank-child.pid"
    ctx, sink = worker(FAKE_CHILD_PIDFILE=str(pidfile))
    with trigger(lambda: bool(sink.getvalue()), ctx.cancel.set):
        assert job.WorkerJob(ctx).run() == job.EXIT_CANCELLED
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise AssertionError("the rank group the agent recorded was left running")
