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
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
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


def launcher_source(recipe: dict) -> Path | None:
    """The build-context file the Dockerfile copies to the runtime entrypoint."""
    build = recipe["execution"]["build"]
    target = recipe["runtime"]["entrypoint"][0]
    for line in (ROOT / build["dockerfile"]).read_text().splitlines():
        words = line.split()
        if words[:1] == ["COPY"] and words[-1] == target and "--from" not in line:
            return ROOT / build["context"]["path"] / words[-2]
    return None


def ray_launchers() -> list[tuple[str, dict, Path]]:
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


def ray_poll_commands(recipe: dict, launcher: Path) -> list[list[str]]:
    node_count = recipe["topology"]["node_count"]
    commands: list[list[str]] = []

    def run(command: list[str], *args: object, **kwargs: object):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, f"VONK_ALIVE={node_count}\n", "")

    argv = [
        str(launcher),
        *recipe["runtime"]["entrypoint"][1:],
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
        patch("os.execv", side_effect=Exec),
        patch("subprocess.run", side_effect=run),
        patch("pathlib.Path.is_file", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
        patch("shutil.which", side_effect=lambda name: f"/image/bin/{name}"),
        patch("time.sleep"),
    ):
        try:
            runpy.run_path(str(launcher), run_name="__main__")
        except Exec:
            pass
    return [
        command
        for command in commands
        if "-c" in command and "import ray" in command[command.index("-c") + 1]
    ]


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
                self.assertIn(path.stem, launchers)

    def test_every_ray_launcher_polls_ray_with_its_own_interpreter(self) -> None:
        for slug, recipe, launcher in ray_launchers():
            with self.subTest(recipe=slug, launcher=str(launcher.relative_to(ROOT))):
                polls = ray_poll_commands(recipe, launcher)
                self.assertTrue(polls, "rank 0 never polled the Ray cluster")
                self.assertEqual(
                    {command[0] for command in polls},
                    {sys.executable},
                    "poll Ray with sys.executable, not a hard-coded interpreter "
                    "path the image may not have",
                )


if __name__ == "__main__":
    unittest.main()
