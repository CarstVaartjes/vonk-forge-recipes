"""A Ray launcher polls its cluster with the interpreter that runs it.

Rank 0 starts the Ray head, then runs ``import ray`` in a child Python until
every node has joined. A hard-coded interpreter path that the image lacks
(``/opt/vllm/.venv/bin/python`` or ``/usr/local/bin/python3`` in the GLM 5.3
dual image, whose python3 is ``/usr/bin/python3``) makes every launch exit with
"runtime executable is missing" right after ``ray start --head``. The launcher's
own ``sys.executable`` always exists, and it is the ``python3`` on the image's
PATH that the build-time ``import ray`` check runs under.

Each launcher runs here for real, as rank 0 of a Ray launch, with only process
creation and file probes faked.

A Ray node is alive before its GPU has registered, so the launcher execs vLLM
only once the cluster reports tensor-parallel x pipeline-parallel GPUs, through
the one shared ``ray_ready.py`` helper (``adapters/**/ray_ready.py``, kept
byte-identical in every Ray launcher's build context).
"""

from __future__ import annotations

import contextlib
import json
import os
import runpy
import subprocess
import sys
import unittest
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from documents import recipe as typed_recipe
from repo_root import ROOT
from test_ray_vllm_launchers import build_time_text

PLACEMENT = {
    "VONK_LOCAL_ADDR": "10.0.0.1",
    "VONK_MASTER_ADDR": "10.0.0.1",
    "VONK_MASTER_PORT": "29500",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_IB_HCA": "roce0",
    "NCCL_IB_GID_INDEX": "3",
    "TP_SOCKET_IFNAME": "eth0",
    "GLOO_SOCKET_IFNAME": "eth0",
}


class Exec(Exception):
    pass


@contextlib.contextmanager
def launcher_helpers(launcher: Path) -> Generator[None]:
    """Import the launcher's own ``ray_ready.py``, as the image has it in /opt/vonk/lib."""
    with (
        patch.object(sys, "path", [str(launcher.parent), *sys.path]),
        patch.dict(sys.modules),
    ):
        sys.modules.pop("ray_ready", None)
        yield


def required_gpus(slug: str) -> int:
    parallelism = typed_recipe(slug).topology.parallelism
    return parallelism.tensor * parallelism.pipeline


def launcher_source(recipe: dict[str, Any]) -> Path | None:
    """The build-context file the Dockerfile copies to the runtime entrypoint."""
    build = recipe["execution"]["build"]
    target = recipe["runtime"]["entrypoint"][0]
    for line in (ROOT / build["dockerfile"]).read_text().splitlines():
        words = line.split()
        if words[:1] == ["COPY"] and words[-1] == target and "--from" not in line:
            return ROOT / build["context"]["path"] / words[-2]
    return None


def is_ray_backend(slug: str) -> bool:
    return typed_recipe(slug).topology.parallelism.backend == "ray"


def ray_launchers() -> list[tuple[str, dict[str, Any], Path]]:
    """Every multi-node recipe whose entrypoint can start a Ray head, by content."""
    found = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        if recipe["topology"].get("node_count", 1) < 2:
            continue
        source = launcher_source(recipe)
        if source and source.is_file() and '"--head"' in source.read_text():
            found.append((path.stem, recipe, source))
    return found


def is_poll(command: list[str]) -> bool:
    """A cluster poll: a child interpreter running `import ray`, or `<ray> status`."""
    if "-c" in command:
        return "import ray" in command[command.index("-c") + 1]
    return command[1:2] == ["status"]


def ray_launch(
    slug: str, recipe: dict[str, Any], launcher: Path
) -> tuple[list[list[str]], list[tuple[tuple[str, ...], int]]]:
    """Run rank 0: every command, and every exec with the polls made before it.

    The first poll sees one GPU; later polls see every GPU the recipe places.
    """
    node_count = recipe["topology"]["node_count"]
    parallelism = typed_recipe(slug).topology.parallelism
    required = required_gpus(slug)
    commands: list[list[str]] = []
    execs: list[tuple[tuple[str, ...], int]] = []
    polls: list[list[str]] = []

    def run(command: list[str], *args: object, **kwargs: object):
        commands.append(list(command))
        gpus = required
        if is_poll(command):
            polls.append(command)
            gpus = 1 if len(polls) == 1 else required
        return subprocess.CompletedProcess(
            command,
            0,
            f"VONK_ALIVE={node_count}\nVONK_GPU={gpus}\nResources: 0.0/{gpus}.0 GPU\n",
            "",
        )

    def execv(path: str, argv: tuple[str, ...]) -> None:
        execs.append((tuple(argv), len(polls)))
        raise Exec

    argv = [
        str(launcher),
        *recipe["runtime"]["entrypoint"][1:],
        "--tensor-parallel-size",
        str(parallelism.tensor),
        "--pipeline-parallel-size",
        str(parallelism.pipeline),
        "--nnodes",
        str(node_count),
        "--node-rank",
        "0",
        "--distributed-executor-backend",
        "ray",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.dict(os.environ, PLACEMENT),
        patch("os.execv", side_effect=execv),
        patch("subprocess.run", side_effect=run),
        patch("subprocess.Popen", return_value=SimpleNamespace(poll=lambda: None)),
        patch("pathlib.Path.is_file", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
        patch("shutil.which", side_effect=lambda name: f"/image/bin/{name}"),
        patch("time.sleep"),
        launcher_helpers(launcher),
        contextlib.suppress(Exec),
    ):
        runpy.run_path(str(launcher), run_name="__main__")
    return commands, execs


def ray_commands(slug: str, recipe: dict[str, Any], launcher: Path) -> list[list[str]]:
    return ray_launch(slug, recipe, launcher)[0]


def ray_poll_commands(commands: list[list[str]]) -> list[list[str]]:
    return [
        command
        for command in commands
        if "-c" in command and "import ray" in command[command.index("-c") + 1]
    ]


def ray_status_commands(commands: list[list[str]]) -> list[list[str]]:
    """The `<ray> status` polls of a mentat-shim launcher."""
    return [command for command in commands if command[1:2] == ["status"]]


class RayPollInterpreterTest(unittest.TestCase):
    def test_the_class_covers_every_ray_backend_recipe(self) -> None:
        launchers = {slug for slug, _, _ in ray_launchers()}
        for path in (ROOT / "recipes").glob("*.json"):
            recipe = json.loads(path.read_text())
            topology = recipe["topology"]
            if (
                topology.get("node_count", 1) > 1
                and topology["parallelism"]["backend"] == "ray"
            ):
                assert path.stem in launchers

    def test_every_ray_launcher_polls_ray_with_its_own_interpreter(self) -> None:
        for slug, recipe, launcher in ray_launchers():
            with self.subTest(recipe=slug, launcher=str(launcher.relative_to(ROOT))):
                commands = ray_commands(slug, recipe, launcher)
                polls = ray_poll_commands(commands)
                statuses = ray_status_commands(commands)
                assert polls or statuses, "rank 0 never polled the Ray cluster"
                # A launcher that polls with the CLI (a mentat shim: `ray status`)
                # needs no interpreter, but the executable it names must be the one
                # the image build checks.
                for command in statuses:
                    assert command[0] in build_time_text(recipe), (
                        "poll Ray with an executable the image build checks"
                    )
                if polls:
                    assert {command[0] for command in polls} == {sys.executable}, (
                        "poll Ray with sys.executable, not a hard-coded interpreter "
                        "path the image may not have"
                    )

    def test_vllm_starts_only_after_every_gpu_registered(self) -> None:
        launchers = [item for item in ray_launchers() if is_ray_backend(item[0])]
        assert len(launchers) == 12
        for slug, recipe, launcher in launchers:
            with self.subTest(recipe=slug):
                assert required_gpus(slug) >= 2
                commands, execs = ray_launch(slug, recipe, launcher)
                assert len(execs) == 1, "rank 0 never started vLLM"
                argv, polls_before = execs[0]
                # The first poll saw one GPU; vLLM waited for the second.
                assert polls_before == 2
                for placement in ("--nnodes", "--node-rank", "--headless"):
                    assert placement not in argv
                starts = [c for c in commands if c[1:2] == ["start"]]
                assert starts
                for command in starts:
                    assert (
                        "--node-ip-address" in command
                        and command[command.index("--node-ip-address") + 1]
                        == PLACEMENT["VONK_LOCAL_ADDR"]
                    ) or f"--node-ip-address={PLACEMENT['VONK_LOCAL_ADDR']}" in command

    def test_every_ray_launcher_ships_the_one_shared_helper(self) -> None:
        copies = {
            path: path.read_bytes()
            for path in (ROOT / "adapters").rglob("ray_ready.py")
        }
        assert len(set(copies.values())) == 1
        for slug, recipe, launcher in ray_launchers():
            if not is_ray_backend(slug):
                continue
            with self.subTest(recipe=slug):
                assert launcher.parent / "ray_ready.py" in copies
                dockerfile = (
                    ROOT / recipe["execution"]["build"]["dockerfile"]
                ).read_text()
                assert "COPY ray_ready.py /opt/vonk/lib/ray_ready.py" in dockerfile


if __name__ == "__main__":
    unittest.main()
