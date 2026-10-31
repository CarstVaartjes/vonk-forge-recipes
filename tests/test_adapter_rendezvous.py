"""Single-node launchers must not depend on container hostname routing."""

from __future__ import annotations

import re
import socket
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.documents import recipe

ROOT = Path(__file__).resolve().parents[1]
MOVA = "video/mova-pytorch/run.py"
CALIBRATION = (
    "deepseek/drowzeys-tr3-hybrid-four/serve/hybrid-port/"
    "b12x/comm/pcie/plain_allreduce_calibration.py"
)
LAUNCH = re.compile(
    r"torch\.distributed\.(?:run|launch)|torchrun\s+--|accelerate\s+launch|deepspeed\s+--"
)


def _rendezvous_errors(source: str) -> list[str]:
    errors: list[str] = []
    if "--standalone" in source:
        errors.append("standalone selects a hostname-based worker store")
    normalized = re.sub(r"[\"',]", " ", source)
    distributed = "init_process_group" in source or LAUNCH.search(normalized)
    if distributed and re.search(
        r"gethostname|getfqdn|tcp://(?!127\.0\.0\.1:|\[::1\]:)", source
    ):
        errors.append("process-group rendezvous must not discover a hostname")
    if not LAUNCH.search(normalized):
        return errors
    # These are local launchers, including the multi-GPU calibration probes.
    # Multi-node vLLM recipes are checked by their recipe topology instead
    # (test_multi_node_vllm_recipes_bind_the_controller_placement); other new
    # multi-node launchers need a separate, explicitly reviewed policy.
    if "--nnodes=1" not in source:
        errors.append("launcher must explicitly select one node")
    if "--rdzv-endpoint=127.0.0.1:" not in source:
        errors.append("rendezvous endpoint must be numeric loopback")
    if "--rdzv-backend=static" in source:
        if not re.search(r"--rdzv-conf=timeout=(?:[1-9]|[12][0-9]|30)\b", source):
            errors.append("static rendezvous needs a timeout of at most 30 seconds")
    elif "--rdzv-backend=c10d" in source:
        if "--local-addr=127.0.0.1" not in source:
            errors.append("c10d worker store must also use numeric loopback")
        for setting in ("join_timeout", "read_timeout"):
            if not re.search(rf"{setting}=(?:[1-9]|[12][0-9]|30)\b", source):
                errors.append(f"c10d needs a short {setting}")
    else:
        errors.append("launcher needs an explicit rendezvous backend")
    return errors


def test_adapter_sources_use_loopback_rendezvous() -> None:
    violations: list[str] = []
    sources = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "adapters"], cwd=ROOT, text=True
    )
    for name in sources.split("\0"):
        path = ROOT / name
        if not path.is_file() or path.suffix not in {"", ".py", ".sh", ".md", ".patch"}:
            continue
        source = path.read_text(encoding="utf-8")
        for error in _rendezvous_errors(source):
            violations.append(f"{path.relative_to(ROOT)}: {error}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "torchrun --standalone --nnodes=1",
        "dist.init_process_group(init_method='tcp://container-name:29500')",
        'os.environ["MASTER_ADDR"] = socket.gethostname(); dist.init_process_group()',
        (
            "torchrun --nnodes=1 --rdzv-backend=static "
            "--rdzv-endpoint=container-name:29500 --rdzv-conf=timeout=10"
        ),
        (
            "torchrun --nnodes=1 --rdzv-backend=c10d "
            "--rdzv-endpoint=127.0.0.1:0 --rdzv-conf=join_timeout=10,read_timeout=10"
        ),
        (
            "torchrun --nnodes=1 --rdzv-backend=static "
            "--rdzv-endpoint=127.0.0.1:29500 --rdzv-conf=timeout=300"
        ),
    ],
)
def test_guard_rejects_unsafe_or_unbounded_launches(source: str) -> None:
    assert _rendezvous_errors(source)


def _capture_launch(adapter: str, tmp_path: Path) -> list[str]:
    path = ROOT / "adapters" / adapter
    runner = types.ModuleType("rendezvous_runner")
    exec(compile(path.read_bytes(), str(path), "exec"), runner.__dict__)  # noqa: S102
    launch = mock.Mock(side_effect=RuntimeError("captured launch"))
    if adapter == MOVA:
        official = tmp_path / "inference.py"
        official.touch()
        with (
            mock.patch.dict(runner.os.environ, {"VONK_JOB_TIMEOUT_SECONDS": "60"}),
            mock.patch.object(
                runner,
                "_arguments",
                return_value=SimpleNamespace(
                    output_dir=tmp_path, seed=0, num_frames=193, num_inference_steps=50
                ),
            ),
            mock.patch.object(runner, "_input_manifest", return_value=None),
            mock.patch.object(runner, "_prompt", return_value="A lake"),
            mock.patch.object(runner, "_variant", return_value=(352, 640)),
            mock.patch.object(runner, "_OFFICIAL_RUNNER", official),
            mock.patch.object(
                runner, "_reference_frame", return_value=tmp_path / "ref.png"
            ),
            mock.patch.object(runner.subprocess, "Popen", launch),
            pytest.raises(RuntimeError, match="captured launch"),
        ):
            runner.main()
    else:
        args = SimpleNamespace(
            world_size=2,
            hidden_size=64,
            rows=(1,),
            dtype="float16",
            warmup=0,
            repeat=1,
            ops_per_sample=1,
            trials=1,
            stabilization_replays=1,
            min_absolute_margin_us=1,
            min_relative_margin=0.1,
            visible_devices="",
        )
        with (
            mock.patch.object(runner.subprocess, "Popen", launch),
            pytest.raises(RuntimeError, match="captured launch"),
        ):
            runner._run_probe(args, tmp_path / "probe.json")
    return launch.call_args.args[0]


@pytest.mark.parametrize("adapter", [MOVA, CALIBRATION])
def test_launcher_uses_available_loopback_port_without_hostname_lookup(
    adapter: str, tmp_path: Path
) -> None:
    with (
        mock.patch.object(
            socket, "gethostname", side_effect=OSError("sandbox hostname")
        ),
        mock.patch.object(socket, "getfqdn", side_effect=OSError("sandbox hostname")),
    ):
        command = _capture_launch(adapter, tmp_path)
        assert "--standalone" not in command
        assert "--rdzv-backend=static" in command
        assert "--rdzv-conf=timeout=10" in command
        assert "--nnodes=1" in command
        assert "--node-rank=0" in command
        assert "--max-restarts=0" in command
        endpoint = next(arg for arg in command if arg.startswith("--rdzv-endpoint="))
        address, port = endpoint.split("=", 1)[1].split(":")
        assert address == "127.0.0.1"
        assert 0 < int(port) < 65536
        # Port selection releases its socket; another launch avoids an occupied port.
        with socket.socket() as listener:
            listener.bind((address, int(port)))
            second = _capture_launch(adapter, tmp_path)
            assert endpoint not in second


PLACEMENT = ("VONK_MASTER_ADDR", "VONK_MASTER_PORT")
LLAMA_405B = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-dual"


def _binds_placement(slug: str) -> bool:
    """Whether the recipe's launcher or runtime names the Controller's rendezvous."""
    document = recipe(slug)
    text = "".join(
        str(value)
        for entry in (*document.runtime.arguments, *document.runtime.environment)
        for value in entry.model_dump().values()
    )
    build = document.execution.build
    if build is not None:
        for path in sorted((ROOT / build.context.path).rglob("*")):
            if path.is_file():
                text += path.read_text(encoding="utf-8", errors="ignore")
    return all(name in text for name in PLACEMENT)


def test_multi_node_vllm_recipes_bind_the_controller_placement() -> None:
    # The platform renders --nnodes/--node-rank/--headless but passes the
    # rendezvous address only as VONK_MASTER_ADDR/VONK_MASTER_PORT, so a launcher
    # that never reads them leaves rank 0 waiting for a client that cannot join.
    unbound = [
        path.stem
        for path in sorted((ROOT / "recipes").glob("*.json"))
        if recipe(path.stem).topology.node_count >= 2
        and recipe(path.stem).runtime.engine == "vllm"
        and not _binds_placement(path.stem)
    ]
    assert not unbound, f"multi-node vLLM recipes without a rendezvous bind: {unbound}"


def _run_llama_405b_wrapper(tmp_path: Path, env: dict[str, str]):
    stub = tmp_path / "vllm"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$VLLM_HOST_IP" "$@"\n')
    stub.chmod(0o755)
    wrapper = ROOT / "adapters/nvidia" / LLAMA_405B / "vllm"
    argv = ["serve", "/models", "--nnodes", "2", "--node-rank", "1", "--headless"]
    result = subprocess.run(
        ["/bin/sh", str(wrapper), *argv],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=False,
    )
    return argv, result


def test_llama_405b_wrapper_binds_the_controller_placement(tmp_path: Path) -> None:
    argv, result = _run_llama_405b_wrapper(
        tmp_path,
        {
            "VONK_LOCAL_ADDR": "192.0.2.2",
            "VONK_MASTER_ADDR": "192.0.2.1",
            "VONK_MASTER_PORT": "29500",
        },
    )
    assert result.returncode == 0, result.stderr
    host, *seen = result.stdout.splitlines()
    assert host == "192.0.2.2"
    assert seen == [*argv, "--master-addr", "192.0.2.1", "--master-port", "29500"]


def test_llama_405b_wrapper_refuses_to_launch_without_placement(
    tmp_path: Path,
) -> None:
    _, result = _run_llama_405b_wrapper(tmp_path, {"VONK_LOCAL_ADDR": "192.0.2.2"})
    assert result.returncode != 0
    assert "Controller rendezvous required" in result.stderr
    assert result.stdout == ""
