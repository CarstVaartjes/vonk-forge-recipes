#!/usr/bin/env python3
"""MiniMax H3 on two Sparks as one video job (vLLM-Omni, creator knapcio/MiniMax-H3-2x-DGX-Spark-TP2).

The platform starts one container per rank and gives each its place through the
environment (VONK_RANK, VONK_WORLD_SIZE, VONK_LOCAL_ADDR, VONK_MASTER_ADDR,
VONK_MASTER_PORT, VONK_PEER_ADDRS).
This one file is both ranks; each rank is an explicit state machine.

Rank 0 (owner)   MAPPED -> AWAIT_AGENT -> SERVING -> HEALTHY -> RENDERING -> WRITTEN -> STOPPING -> DONE
Rank 1 (worker)  MAPPED -> AGENT -> STOPPING -> DONE

A failure or a SIGTERM from any state goes to STOPPING; the outcome decides the
exit code (0 done, 1 failed, 143 cancelled). Rank 0 runs the creator's serve line
(scripts/start-head.sh) and posts one request shaped like scripts/render.sh.
Rank 1 runs the creator's agent (h3_tp2.agent, port 18095) and streams the
rank-1 log to stdout. Both print a heartbeat sized to the submitter's timeout,
because the platform's inactivity window is the only liveness it has.
"""

from __future__ import annotations

import argparse
import enum
import http.client
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

VENV = Path("/opt/vonk/venv")
INPUT_ROOT = Path("/inputs")
AGENT_PORT = 18095  # the creator's default
OUTPUT_NAME = "video.mp4"
PROMPT_MAX_BYTES = 16 * 1024
REQUEST_MAX_BYTES = 64 * 1024
IMAGE_MAX_BYTES = 30 * 1024 * 1024  # vLLM-Omni's reference image limit
IMAGE_SUFFIXES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
ASPECT_RATIO = "16:9"  # what the creator's render.sh always sends; width and height decide the canvas
REF2VA_MAX_IMAGES = 9  # vLLM-Omni: ref2va accepts at most 9 image references
AUDIO_FLOW_SHIFT = 3.0  # the creator's render.sh default
EXIT_DONE, EXIT_FAILED, EXIT_CONFIG, EXIT_CANCELLED = 0, 1, 2, 143


class ConfigError(Exception):
    """The job cannot start: a variable, argument or input is missing or invalid."""


class JobFailure(Exception):
    """A running job ended without its video."""


class Cancelled(Exception):
    """The platform asked this rank to stop (SIGTERM)."""


class Outcome(enum.Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def wait(self, cancel: threading.Event, seconds: float) -> bool:
        """Sleep up to ``seconds`` or until cancelled; True when cancelled."""


class RealClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def wait(self, cancel: threading.Event, seconds: float) -> bool:
        return cancel.wait(max(0.0, seconds))


@dataclass(frozen=True)
class Timing:
    """Every wait has its own bound; the defaults are the design's, tests shrink them."""

    agent_ready_s: float = 3600.0  # H3_TP2_WORKER_READY_TIMEOUT
    probe_first_s: float = 1.0
    probe_cap_s: float = 15.0
    health_cap_s: float = 3900.0  # the server's own --init-timeout 3600 plus margin
    health_poll_s: float = 5.0
    request_s: float = 14400.0  # H3_TP2_REQUEST_TIMEOUT_S
    request_margin_s: float = 600.0
    render_poll_s: float = 1.0
    server_grace_s: float = 60.0
    agent_grace_s: float = 30.0
    worker_grace_s: float = 15.0
    tail_poll_s: float = 0.5


class MachineState(enum.Enum):
    pass


class HeadState(MachineState):
    MAPPED = "MAPPED"
    AWAIT_AGENT = "AWAIT_AGENT"
    SERVING = "SERVING"
    HEALTHY = "HEALTHY"
    RENDERING = "RENDERING"
    WRITTEN = "WRITTEN"
    STOPPING = "STOPPING"
    DONE = "DONE"


class WorkerState(MachineState):
    MAPPED = "MAPPED"
    AGENT = "AGENT"
    STOPPING = "STOPPING"
    DONE = "DONE"


_H = HeadState
HEAD_EDGES: Mapping[HeadState, frozenset[HeadState]] = {
    _H.MAPPED: frozenset({_H.AWAIT_AGENT, _H.STOPPING}),
    _H.AWAIT_AGENT: frozenset({_H.SERVING, _H.STOPPING}),
    _H.SERVING: frozenset({_H.HEALTHY, _H.STOPPING}),
    _H.HEALTHY: frozenset({_H.RENDERING, _H.STOPPING}),
    _H.RENDERING: frozenset({_H.WRITTEN, _H.STOPPING}),
    _H.WRITTEN: frozenset({_H.STOPPING}),
    _H.STOPPING: frozenset({_H.DONE}),
    _H.DONE: frozenset(),
}
_W = WorkerState
WORKER_EDGES: Mapping[WorkerState, frozenset[WorkerState]] = {
    _W.MAPPED: frozenset({_W.AGENT, _W.STOPPING}),
    _W.AGENT: frozenset({_W.STOPPING}),
    _W.STOPPING: frozenset({_W.DONE}),
    _W.DONE: frozenset(),
}


class Machine:
    """The one place a rank changes state: ``step`` checks the edge and logs it."""

    def __init__(self, rank: int, edges: Mapping[Any, frozenset[Any]], start: MachineState, clock: Clock) -> None:
        self.rank = rank
        self.state = start
        self.outcome = Outcome.RUNNING
        self.reason = ""
        self._edges = edges
        self._clock = clock
        self._started = clock.monotonic()

    def elapsed(self) -> float:
        return self._clock.monotonic() - self._started

    def step(self, to: MachineState, note: str = "") -> None:
        if to not in self._edges[self.state]:
            raise RuntimeError(f"rank {self.rank}: no edge {self.state.value} -> {to.value}")
        say(f"rank {self.rank} {self.state.value} -> {to.value}" + (f" ({note})" if note else ""))
        self.state = to

    def settle(self, outcome: Outcome, reason: str = "") -> None:
        if self.outcome is Outcome.RUNNING:
            self.outcome = outcome
            self.reason = reason

    def exit_code(self) -> int:
        return {
            Outcome.DONE: EXIT_DONE,
            Outcome.FAILED: EXIT_FAILED,
            Outcome.CANCELLED: EXIT_CANCELLED,
            Outcome.RUNNING: EXIT_FAILED,
        }[self.outcome]


def say(message: str) -> None:
    print(f"vllm-omni-job: {message}", flush=True)


# --- placement and environment --------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    rank: int
    local: str
    master: str
    master_port: int
    peers: tuple[str, ...]
    job_timeout_s: int | None

    @property
    def worker(self) -> str:
        return self.peers[1]


def _require(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set")
    return value


def read_placement(env: Mapping[str, str]) -> Placement:
    """Map the platform's VONK_* environment. Errors name the variable, never its value."""
    if _require(env, "VONK_WORLD_SIZE") != "2":
        raise ConfigError("VONK_WORLD_SIZE must be 2: this recipe runs on two Sparks")
    rank = _require(env, "VONK_RANK")
    if rank not in ("0", "1"):
        raise ConfigError("VONK_RANK must be 0 or 1")
    local = _require(env, "VONK_LOCAL_ADDR")
    master = _require(env, "VONK_MASTER_ADDR")
    port_text = _require(env, "VONK_MASTER_PORT")
    if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
        raise ConfigError("VONK_MASTER_PORT must be a port number")
    peers = tuple(part.strip() for part in _require(env, "VONK_PEER_ADDRS").split(","))
    if len(peers) != 2 or not all(peers):
        raise ConfigError("VONK_PEER_ADDRS must list the two ranks' addresses in rank order")
    timeout = env.get("VONK_JOB_TIMEOUT_SECONDS", "").strip()
    return Placement(
        rank=int(rank),
        local=local,
        master=master,
        master_port=int(port_text),
        peers=peers,
        job_timeout_s=int(timeout) if timeout.isdigit() else None,
    )


def heartbeat_interval(job_timeout_s: int | None) -> float:
    """Quarter of the submitter's inactivity window, within 5..60 s (30 s when unknown)."""
    if job_timeout_s is None:
        return 30.0
    return float(max(5, min(60, job_timeout_s // 4)))


def child_environment(place: Placement, base: Mapping[str, str]) -> dict[str, str]:
    """The creator's H3_TP2_* variables (scripts/env.sh) from the platform's placement."""
    env = dict(base)
    env.pop("H3_TP2_SECRET", None)  # unset: the creator's channel then runs without HMAC, as it allows
    env.update(
        H3_TP2_HEAD_IP=place.master,
        H3_TP2_WORKER_IP=place.worker,
        H3_TP2_WORKER_AGENT=f"{place.worker}:{AGENT_PORT}",
        H3_TP2_MASTER_ADDR=place.master,
        H3_TP2_WORLD_SIZE="2",
        VLLM_HOST_IP=place.local,
    )
    if place.rank == 0:
        env.update(H3_TP2_ROLE="head", MASTER_PORT=str(place.master_port))
    return env


def serve_argv(model: str, port: int) -> list[str]:
    """The creator's serve line (scripts/start-head.sh), with the API on loopback."""
    return [
        str(VENV / "bin" / "vllm"),
        "serve",
        model,
        "--omni",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--init-timeout",
        "3600",
        "--num-gpus",
        "2",
        "--tensor-parallel-size",
        "1",
        "--text-encoder-tp-size",
        "2",
        "--usp",
        "2",
        "--ring",
        "1",
        "--vae-patch-parallel-size",
        "2",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--quantization",
        "fp8",
        "--enforce-eager",
        "--diffusion-attention-backend",
        "CUDNN_ATTN",
        "--distributed-executor-backend",
        "h3_tp2.executor.RemoteReaderExecutor",
    ]


def agent_argv(place: Placement, log_dir: Path) -> list[str]:
    """The creator's agent (scripts/start-agent.sh) on this rank's fabric address."""
    python = str(VENV / "bin" / "python")
    return [
        python,
        "-m",
        "h3_tp2.agent",
        "--bind",
        f"{place.local}:{AGENT_PORT}",
        "--python",
        python,
        "--log-dir",
        str(log_dir),
    ]


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --- the request ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Defaults and bounds from the recipe's arguments; request.json may override the values."""

    task: str  # "fl2va" (FL2VA partition: t2va or fl2va) or "ref2va"
    model: str
    duration: float
    min_duration: float
    max_duration: float
    width: int
    height: int
    steps: int
    flow_shift: float
    seed: int


@dataclass(frozen=True)
class RenderRequest:
    fields: tuple[tuple[str, str], ...]
    images: tuple[Path, ...]
    image_field: str  # "input_reference" for one image, "input_references" for several

    def field(self, name: str) -> str:
        return dict(self.fields)[name]


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low:g} and {high:g}")
    return float(value)


def _integer(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ConfigError(f"{name} must be an integer between {low} and {high}")
    return value


def _geometry(value: object, name: str) -> int:
    number = _integer(value, name, 64, 2048)
    if number % 32:
        raise ConfigError(f"{name} must be a multiple of 32")
    return number


def _input_file(inputs: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ConfigError(f"{name} must be a non-empty relative input path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ConfigError(f"{name} must stay inside the input directory")
    candidate = inputs / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ConfigError(f"{name} does not name a regular input file")
    try:
        candidate.resolve(strict=True).relative_to(inputs.resolve(strict=True))
    except ValueError:
        raise ConfigError(f"{name} escapes the input directory") from None
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        raise ConfigError(f"{name} must be a JPEG, PNG or WebP image")
    if not 0 < candidate.stat().st_size <= IMAGE_MAX_BYTES:
        raise ConfigError(f"{name} must be a non-empty image of at most 30 MiB")
    return candidate


def _read_prompt(inputs: Path) -> str:
    prompts = sorted(path for path in inputs.iterdir() if path.suffix.lower() == ".txt")
    if len(prompts) != 1:
        raise ConfigError("exactly one UTF-8 .txt prompt file is required")
    path = prompts[0]
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= PROMPT_MAX_BYTES:
        raise ConfigError("the prompt file must hold 1..16384 UTF-8 bytes")
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise ConfigError("the prompt file must be valid UTF-8") from None
    if not text.strip() or "\x00" in text:
        raise ConfigError("the prompt must be non-empty text")
    return text  # byte for byte, as the creator's client sends it


def _read_request(inputs: Path) -> dict[str, Any]:
    path = inputs / "request.json"
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > REQUEST_MAX_BYTES:
        raise ConfigError("request.json must be a regular file of at most 64 KiB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ConfigError("request.json must contain UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise ConfigError("request.json must be a JSON object")
    return value


def _json_number(value: float) -> str:
    return json.dumps(int(value) if float(value).is_integer() else value)


def build_request(settings: Settings, inputs: Path = INPUT_ROOT) -> RenderRequest:
    """Validate the inputs and shape them like the creator's scripts/render.sh."""
    common = {"duration", "width", "height", "num_inference_steps", "flow_shift", "seed"}
    allowed = common | ({"references"} if settings.task == "ref2va" else {"first_image", "last_image"})
    if not inputs.is_dir() or inputs.is_symlink():
        raise ConfigError("the input directory is missing")
    request = _read_request(inputs)
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ConfigError(f"request.json has unsupported fields: {', '.join(unknown)}")
    prompt = _read_prompt(inputs)
    duration = _number(
        request.get("duration", settings.duration), "duration", settings.min_duration, settings.max_duration
    )
    width = _geometry(request.get("width", settings.width), "width")
    height = _geometry(request.get("height", settings.height), "height")
    steps = _integer(request.get("num_inference_steps", settings.steps), "num_inference_steps", 1, 50)
    shift = _number(request.get("flow_shift", settings.flow_shift), "flow_shift", 1e-6, 1e6)
    seed = _integer(request.get("seed", settings.seed), "seed", 0, 2**63 - 1)

    extra: dict[str, Any] = {
        "task": settings.task,
        "duration": int(duration) if duration.is_integer() else duration,
        "audio_flow_shift": AUDIO_FLOW_SHIFT,
    }
    if settings.task == "ref2va":
        refs = request.get("references")
        if not isinstance(refs, list) or not 1 <= len(refs) <= REF2VA_MAX_IMAGES:
            raise ConfigError(f"references must list 1..{REF2VA_MAX_IMAGES} relative image paths")
        images = tuple(_input_file(inputs, ref, "references") for ref in refs)
        extra.update(long_video=True, long_video_mode="continuation")
    else:
        first = request.get("first_image")
        last = request.get("last_image")
        images = tuple(
            _input_file(inputs, value, name)
            for name, value in (("first_image", first), ("last_image", last))
            if value is not None
        )
        if images:
            extra["task"] = "fl2va"
            extra["frame_indices"] = [0] if last is None else [-1] if first is None else [0, -1]
        else:
            extra["task"] = "t2va"
    fields = (
        ("prompt", prompt),
        ("width", str(width)),
        ("height", str(height)),
        ("aspect_ratio", ASPECT_RATIO),
        ("fps", "24"),
        ("num_inference_steps", str(steps)),
        ("flow_shift", _json_number(shift)),
        ("seed", str(seed)),
        ("extra_params", json.dumps(extra, ensure_ascii=False)),
    )
    return RenderRequest(fields, images, "input_references" if len(images) > 1 else "input_reference")


def encode_multipart(request: RenderRequest) -> tuple[bytes, str]:
    boundary = f"vonk{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in request.fields:
        parts.append(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n').encode() + value.encode() + b"\r\n"
        )
    for image in request.images:
        media = IMAGE_SUFFIXES[image.suffix.lower()]
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{request.image_field}"; '
            f'filename="{image.name}"\r\nContent-Type: {media}\r\n\r\n'
        )
        parts.append(head.encode() + image.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# --- processes ------------------------------------------------------------------------------


def stop_group(proc: subprocess.Popen[bytes], grace_s: float, clock: Clock) -> None:
    """SIGTERM the process group, SIGKILL it after the grace period."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    gone = threading.Event()
    deadline = clock.monotonic() + grace_s
    while proc.poll() is None and clock.monotonic() < deadline:
        clock.wait(gone, 0.1)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def spawn(argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(argv),
        env=dict(env),
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


class Heartbeat:
    """Prints ``<state> <elapsed>s`` until stopped, so the platform sees the rank is alive."""

    def __init__(self, machine: Machine, interval_s: float) -> None:
        self._machine = machine
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            say(f"rank {self._machine.rank} {self._machine.state.value} {self._machine.elapsed():.0f}s")

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


@dataclass
class Context:
    place: Placement
    settings: Settings | None
    env: Mapping[str, str]
    output_dir: Path
    cancel: threading.Event = field(default_factory=threading.Event)
    clock: Clock = field(default_factory=RealClock)
    timing: Timing = field(default_factory=Timing)
    request: RenderRequest | None = None  # rank 0: validated once, by main, before any model work
    heartbeat_s: float | None = None
    log_out: Any = None  # where rank 1 copies the agent's log; stdout when None

    def interval(self) -> float:
        return self.heartbeat_s if self.heartbeat_s is not None else heartbeat_interval(self.place.job_timeout_s)


def _env_seconds(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(env.get(name, ""))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def timing_from_environment(env: Mapping[str, str], base: Timing | None = None) -> Timing:
    """The creator's timeouts (H3_TP2_*) travel as recipe environment; the wrapper reads them."""
    base = base or Timing()
    return replace(
        base,
        agent_ready_s=_env_seconds(env, "H3_TP2_WORKER_READY_TIMEOUT", base.agent_ready_s),
        request_s=_env_seconds(env, "H3_TP2_REQUEST_TIMEOUT_S", base.request_s),
    )


# --- rank 0 ---------------------------------------------------------------------------------


class RenderCall(threading.Thread):
    """One POST /v1/videos/sync on a thread, so the state machine can watch the server and SIGTERM."""

    def __init__(self, port: int, body: bytes, content_type: str, target: Path, read_timeout_s: float) -> None:
        super().__init__(name="render", daemon=True)
        self._port = port
        self._body = body
        self._content_type = content_type
        self._target = target
        self._read_timeout = read_timeout_s
        self.error = ""
        self.bytes = 0

    def run(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self._port, timeout=self._read_timeout)
        try:
            conn.request("POST", "/v1/videos/sync", body=self._body, headers={"Content-Type": self._content_type})
            response = conn.getresponse()
            if response.status != 200:
                detail = response.read(2048).decode("utf-8", "replace").strip()
                self.error = f"the server answered HTTP {response.status}: {detail}"
                return
            with self._target.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    self.bytes += len(chunk)
        except (OSError, http.client.HTTPException) as exc:
            self.error = f"the render request failed: {type(exc).__name__}: {exc}"
        finally:
            conn.close()


class HeadJob:
    def __init__(self, ctx: Context) -> None:
        assert ctx.settings is not None
        self.ctx = ctx
        self.settings = ctx.settings
        self.machine = Machine(0, HEAD_EDGES, HeadState.MAPPED, ctx.clock)
        self.server: subprocess.Popen[bytes] | None = None
        self.port = 0

    # Bounded waits: each returns when its condition holds, raises when its bound or the server fails.
    def _check(self) -> None:
        if self.ctx.cancel.is_set():
            raise Cancelled()

    def _sleep(self, seconds: float) -> None:
        self.ctx.clock.wait(self.ctx.cancel, seconds)
        self._check()

    def _await_agent(self) -> None:
        timing, clock = self.ctx.timing, self.ctx.clock
        deadline = clock.monotonic() + timing.agent_ready_s
        delay = timing.probe_first_s
        target = (self.ctx.place.worker, AGENT_PORT)
        while True:
            self._check()
            try:
                socket.create_connection(target, timeout=5).close()
                return
            except OSError:
                pass
            if clock.monotonic() + delay > deadline:
                raise JobFailure(f"the worker agent on port {AGENT_PORT} did not answer within {timing.agent_ready_s:g} s")
            self._sleep(delay)
            delay = min(delay * 2, timing.probe_cap_s)

    def _health_ok(self) -> bool:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", "/health")
            return conn.getresponse().status == 200
        except (OSError, http.client.HTTPException):
            return False
        finally:
            conn.close()

    def _await_health(self) -> None:
        assert self.server is not None
        timing, clock = self.ctx.timing, self.ctx.clock
        deadline = clock.monotonic() + timing.health_cap_s
        while True:
            self._check()
            code = self.server.poll()
            if code is not None:
                raise JobFailure(f"the server exited with status {code} before it was healthy")
            if self._health_ok():
                return
            if clock.monotonic() >= deadline:
                raise JobFailure(f"the server was not healthy within {timing.health_cap_s:g} s")
            self._sleep(timing.health_poll_s)

    def _render(self, scratch: Path) -> Path:
        assert self.server is not None
        assert self.ctx.request is not None
        body, content_type = encode_multipart(self.ctx.request)
        partial = scratch / "video.partial.mp4"
        timing = self.ctx.timing
        call = RenderCall(self.port, body, content_type, partial, timing.request_s + timing.request_margin_s)
        self.machine.step(HeadState.RENDERING, "POST /v1/videos/sync")
        call.start()
        while call.is_alive():
            call.join(timing.render_poll_s)
            self._check()
            code = self.server.poll()
            if call.is_alive() and code is not None:
                raise JobFailure(f"the server exited with status {code} during the render")
        if call.error:
            raise JobFailure(call.error)
        with partial.open("rb") as written:
            head = written.read(12)
        if call.bytes <= 0 or head[4:8] != b"ftyp":
            raise JobFailure("the server returned a body that is not an MP4 file")
        return partial

    def _publish(self, partial: Path) -> None:
        os.replace(partial, self.ctx.output_dir / OUTPUT_NAME)  # same filesystem: atomic
        self.machine.step(HeadState.WRITTEN, OUTPUT_NAME)

    def run(self) -> int:
        ctx = self.ctx
        try:
            with Heartbeat(self.machine, ctx.interval()):
                scratch = Path(tempfile.mkdtemp(prefix="vllm-omni-"))
                try:
                    self._drive(scratch)
                except Cancelled:
                    self.machine.settle(Outcome.CANCELLED, "stopped by the platform")
                except (JobFailure, ConfigError) as exc:
                    self.machine.settle(Outcome.FAILED, str(exc))
                finally:
                    self.machine.step(HeadState.STOPPING, self.machine.reason or "finished")
                    self._stop_server()
                    self.machine.step(HeadState.DONE)
                    shutil.rmtree(scratch, ignore_errors=True)
        except OSError as exc:  # scratch or output unwritable
            self.machine.settle(Outcome.FAILED, f"{type(exc).__name__}: {exc}")
        self.machine.settle(Outcome.DONE)
        if self.machine.outcome is Outcome.FAILED:
            say(f"rank 0 failed: {self.machine.reason}")
        return self.machine.exit_code()

    def _drive(self, scratch: Path) -> None:
        ctx = self.ctx
        self.machine.step(HeadState.AWAIT_AGENT, f"{ctx.place.worker}:{AGENT_PORT}")
        self._await_agent()
        self.port = free_loopback_port()
        env = child_environment(ctx.place, ctx.env)
        self.machine.step(HeadState.SERVING, f"vllm serve {self.settings.model} on 127.0.0.1:{self.port}")
        self.server = spawn(serve_argv(self.settings.model, self.port), env, scratch)
        self.machine.step(HeadState.HEALTHY, "waiting for GET /health")
        self._await_health()
        partial = self._render(scratch)
        self._publish(partial)

    def _stop_server(self) -> None:
        if self.server is not None:
            stop_group(self.server, self.ctx.timing.server_grace_s, self.ctx.clock)


# --- rank 1 ---------------------------------------------------------------------------------


@dataclass
class LogTail:
    """Copies new bytes of the newest ``worker-rank1-*.log`` to ``out``; the agent writes rank 1's output there."""

    directory: Path
    out: Any
    path: Path | None = None
    offset: int = 0

    def poll(self) -> int:
        logs = sorted(self.directory.glob("worker-rank1-*.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return 0
        if logs[-1] != self.path:
            self.path, self.offset = logs[-1], 0
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        self.offset += len(data)
        if data:
            self.out.write(data)
            self.out.flush()
        return len(data)


class WorkerJob:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.machine = Machine(1, WORKER_EDGES, WorkerState.MAPPED, ctx.clock)
        self.agent: subprocess.Popen[bytes] | None = None
        self.log_dir: Path | None = None

    def run(self) -> int:
        ctx = self.ctx
        try:
            with Heartbeat(self.machine, ctx.interval()):
                scratch = Path(tempfile.mkdtemp(prefix="vllm-omni-"))
                self.log_dir = scratch / "h3-tp2-logs"
                try:
                    self._serve_agent(scratch)
                except Cancelled:
                    self.machine.settle(Outcome.CANCELLED, "stopped by the platform")
                except JobFailure as exc:
                    self.machine.settle(Outcome.FAILED, str(exc))
                finally:
                    self.machine.step(WorkerState.STOPPING, self.machine.reason or "finished")
                    self._stop()
                    self.machine.step(WorkerState.DONE)
                    shutil.rmtree(scratch, ignore_errors=True)
        except OSError as exc:
            self.machine.settle(Outcome.FAILED, f"{type(exc).__name__}: {exc}")
        self.machine.settle(Outcome.FAILED, "the worker ended without being asked to stop")
        if self.machine.outcome is Outcome.FAILED:
            say(f"rank 1 failed: {self.machine.reason}")
        return self.machine.exit_code()

    def _serve_agent(self, scratch: Path) -> None:
        ctx = self.ctx
        assert self.log_dir is not None
        self.machine.step(WorkerState.AGENT, f"h3_tp2.agent on {ctx.place.local}:{AGENT_PORT}")
        self.agent = spawn(agent_argv(ctx.place, self.log_dir), child_environment(ctx.place, ctx.env), scratch)
        tail = LogTail(self.log_dir, ctx.log_out or sys.stdout.buffer)
        while True:
            if ctx.cancel.is_set():
                raise Cancelled()
            code = self.agent.poll()
            tail.poll()  # also after an exit: its last lines say why
            if code is not None:
                raise JobFailure(f"the agent exited with status {code}")
            ctx.clock.wait(ctx.cancel, ctx.timing.tail_poll_s)

    def _stop(self) -> None:
        timing, clock = self.ctx.timing, self.ctx.clock
        if self.agent is not None:
            stop_group(self.agent, timing.agent_grace_s, clock)
        # The agent starts the rank in its own session and records the group in rank1.pid; end it too.
        if self.log_dir is not None:
            pid_file = self.log_dir / "rank1.pid"
            try:
                group = int(pid_file.read_text().strip())
            except (OSError, ValueError):
                return
            for sig, grace in ((signal.SIGTERM, timing.worker_grace_s), (signal.SIGKILL, 0.0)):
                try:
                    os.killpg(group, sig)
                except (ProcessLookupError, PermissionError):
                    return
                deadline = clock.monotonic() + grace
                gone = threading.Event()
                while clock.monotonic() < deadline and _group_alive(group):
                    clock.wait(gone, 0.1)
                if not _group_alive(group):
                    return


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# --- entry ----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="the partition directory: /models/FL2VA or /models/Ref2VA")
    parser.add_argument("--task", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--min-duration", type=float, required=True)
    parser.add_argument("--max-duration", type=float, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--flow-shift", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings(
        task=args.task,
        model=args.model,
        duration=args.duration,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        width=args.width,
        height=args.height,
        steps=args.num_inference_steps,
        flow_shift=args.flow_shift,
        seed=args.seed,
    )


def install_term_handlers(cancel: threading.Event) -> None:
    def on_term(_signum: int, _frame: object) -> None:
        cancel.set()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    inputs: Path = INPUT_ROOT,
) -> int:
    env = os.environ if env is None else env
    try:
        args = parse_args(argv)
        place = read_placement(env)
        settings = settings_from(args)
        request = build_request(settings, inputs) if place.rank == 0 else None  # a bad request refuses before any model work
    except ConfigError as exc:
        print(f"vllm-omni-job: refusing to start: {exc}", file=sys.stderr, flush=True)
        return EXIT_CONFIG
    ctx = Context(
        place, settings, env, args.output_dir, timing=timing_from_environment(env), request=request
    )
    install_term_handlers(ctx.cancel)
    say(f"rank {place.rank} of 2, local {place.local}, master {place.master}:{place.master_port}")
    return (HeadJob(ctx) if place.rank == 0 else WorkerJob(ctx)).run()


if __name__ == "__main__":
    sys.exit(main())
