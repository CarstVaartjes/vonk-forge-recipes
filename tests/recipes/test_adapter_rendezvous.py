"""Single-node launchers must not depend on container hostname routing."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import runpy
import socket
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from repo_root import ROOT

from tests.documents import recipe, recipe_document
from tests.recipes.test_ray_poll_interpreter import launcher_source

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
        errors.extend(
            f"c10d needs a short {setting}"
            for setting in ("join_timeout", "read_timeout")
            if not re.search(rf"{setting}=(?:[1-9]|[12][0-9]|30)\b", source)
        )
    else:
        errors.append("launcher needs an explicit rendezvous backend")
    return errors


def test_adapter_sources_use_loopback_rendezvous() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "adapters").rglob("*")):
        if not path.is_file() or path.suffix not in {"", ".py", ".sh", ".md", ".patch"}:
            continue
        source = path.read_text(encoding="utf-8")
        violations.extend(
            f"{path.relative_to(ROOT)}: {error}" for error in _rendezvous_errors(source)
        )
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
LLAMA_405B = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-four"


def _binds_placement(slug: str) -> bool:
    """Whether the recipe's launcher or runtime names the Controller's rendezvous."""
    document = recipe(slug)
    text = "".join(
        str(value)
        for entry in (*document.runtime.arguments, *document.runtime.environment)
        for value in entry.model_dump().values()
    )
    build = document.execution.build
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


FABRIC = "192.0.2.77"
FABRIC_ENV = {
    "VONK_LOCAL_ADDR": FABRIC,
    "VONK_MASTER_ADDR": "192.0.2.1",
    "VONK_MASTER_PORT": "29500",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_IB_HCA": "roce0",
    "NCCL_IB_GID_INDEX": "3",
    "TP_SOCKET_IFNAME": "eth0",
    "GLOO_SOCKET_IFNAME": "eth0",
}
# Shell launchers run in their own behavioural test (the llama wrapper above).
SHELL_LAUNCHERS = {LLAMA_405B}


class _Stop(BaseException):
    pass


def _advertised_host(slug: str, home: Path) -> list[str | None]:
    """VLLM_HOST_IP each process spawn of the rank-1 launcher would carry."""
    document = recipe_document(slug)
    launcher = launcher_source(document)
    assert launcher is not None, slug
    seen: list[str | None] = []

    def spawn(*args: object, **kwargs: object) -> None:
        env = kwargs.get("env")
        if env is None and args and isinstance(args[-1], dict):
            env = args[-1]
        seen.append((env if isinstance(env, dict) else os.environ).get("VLLM_HOST_IP"))
        if len(seen) > 50:
            raise _Stop

    def execed(*args: object, **kwargs: object) -> None:
        spawn(*args, **kwargs)
        raise _Stop

    authored: list[str] = []
    for item in document["runtime"]["arguments"]:
        value = item.get("value")
        if value is None and item.get("setting"):
            value = document["settings"][item["setting"]]["value"]
        if value is True:
            authored.append(f"--{item['name']}")
        elif value is not None and value is not False:
            authored.extend((f"--{item['name']}", str(value)))
    argv = [str(launcher), *document["runtime"]["entrypoint"][1:], *authored]
    parallelism = document["topology"]["parallelism"]

    def add(name: str, *value: str) -> None:
        # The Controller appends placement only where the recipe did not author it.
        if f"--{name}" not in argv:
            argv.extend((f"--{name}", *value))

    add("tensor-parallel-size", str(parallelism["tensor"]))
    add("pipeline-parallel-size", str(parallelism["pipeline"]))
    if parallelism["backend"] in {"mp", "ray"}:
        add("distributed-executor-backend", parallelism["backend"])
    add("nnodes", str(document["topology"]["node_count"]))
    add("node-rank", "1")
    if "--headless" not in argv:
        argv.append("--headless")
    environment = {k: v for k, v in os.environ.items() if k != "VLLM_HOST_IP"}
    ok = subprocess.CompletedProcess(
        [], 0, "VONK_ALIVE=9\nResources: 0.0/9.0 GPU\n", ""
    )
    real_open = io.open

    def read_only_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        # Image files the launcher reads (model configs, plans) are empty JSON.
        if set(mode) & set("wax+"):
            return io.StringIO()
        try:
            return real_open(file, mode, *args, **kwargs)  # type: ignore[call-overload]
        except FileNotFoundError:
            return io.BytesIO(b"{}") if "b" in mode else io.StringIO("{}")

    real_read_text = Path.read_text

    def read_text(self: Path, *args: object, **kwargs: object) -> str:
        try:
            return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]
        except FileNotFoundError:
            return "{}"

    patches = [
        mock.patch("builtins.open", read_only_open),
        mock.patch("io.open", read_only_open),
        mock.patch("pathlib.Path.read_text", read_text),
        mock.patch.object(sys, "argv", argv),
        mock.patch.dict(
            os.environ,
            {**environment, **FABRIC_ENV, "HOME": str(home), "TMPDIR": str(home)},
            clear=True,
        ),
        *(
            mock.patch(f"os.{name}", side_effect=execed)
            for name in ("execv", "execve", "execvp", "execvpe")
        ),
        mock.patch(
            "subprocess.run", side_effect=lambda *a, **k: (spawn(*a, **k), ok)[1]
        ),
        mock.patch("subprocess.Popen", side_effect=spawn),
        mock.patch("pathlib.Path.is_file", return_value=True),
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("os.access", return_value=True),
        mock.patch("shutil.which", side_effect=lambda name, **_: f"/image/bin/{name}"),
        mock.patch("time.sleep"),
        # Nothing is written or removed on the host.
        mock.patch("pathlib.Path.mkdir"),
        mock.patch("pathlib.Path.write_text"),
        mock.patch("pathlib.Path.touch"),
        mock.patch("pathlib.Path.symlink_to"),
        mock.patch("pathlib.Path.unlink"),
        mock.patch("os.makedirs"),
        mock.patch("os.symlink"),
        mock.patch("os.remove"),
        mock.patch("os.replace"),
        mock.patch("pathlib.Path.replace"),
        mock.patch("shutil.rmtree"),
        mock.patch("os.unlink"),
        mock.patch("os.rename"),
        mock.patch("os.rmdir"),
        mock.patch("os.chmod"),
        mock.patch("os.system"),
        mock.patch("pathlib.Path.rename"),
        mock.patch("pathlib.Path.rmdir"),
        mock.patch("pathlib.Path.chmod"),
        mock.patch("shutil.copytree"),
        mock.patch("shutil.move"),
        mock.patch("shutil.copy"),
        mock.patch("shutil.copyfile"),
    ]
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        stack.enter_context(contextlib.suppress(_Stop))
        runpy.run_path(str(launcher), run_name="__main__")
    return seen


def _distributed_vllm_slugs() -> list[str]:
    return [
        path.stem
        for path in sorted((ROOT / "recipes").glob("*.json"))
        if recipe(path.stem).topology.node_count >= 2
        and recipe(path.stem).runtime.engine == "vllm"
        and path.stem not in SHELL_LAUNCHERS
    ]


# Launchers whose image-only inputs (an EXL3 plan; derived views built by real tools) the fakes
# cannot stand in for: only their launcher source is checked.
SOURCE_ONLY = {
    "deepseek-v4-1-flash-exl3-0xsero-vllm-dual",
    "glm-5-3-flash-nvfp4-dflash2-tonyd2wild-vllm-dual",
}
FROM_FABRIC = re.compile(
    r"(\w+)\s*=\s*os\.environ(?:\.get\(|\[)\s*[\"']VONK_LOCAL_ADDR[\"'][\s\S]*"
    r"os\.environ\[[\"']VLLM_HOST_IP[\"']\]\s*=\s*\1\b"
)


@pytest.mark.parametrize("slug", _distributed_vllm_slugs())
def test_distributed_vllm_launcher_advertises_the_fabric_address(
    slug: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # vLLM's message queue advertises VLLM_HOST_IP, or else the default-route
    # source address, which is the LAN.
    if slug in SOURCE_ONLY:
        document = recipe_document(slug)
        assert FROM_FABRIC.search(launcher_source(document).read_text())  # type: ignore[union-attr]
        return
    seen = _advertised_host(slug, tmp_path)
    assert seen, "launcher never reached an exec"
    assert seen[-1] == FABRIC  # the exec itself, not an earlier probe
    assert set(seen) <= {FABRIC, None}


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
