"""A vLLM recipe that declares the Ray backend must ship a Ray-aware launcher.

The platform launches one process per rank and appends ``--nnodes``,
``--node-rank`` and (for a worker) ``--headless``. vLLM rejects ``--nnodes`` > 1
with ``--distributed-executor-backend ray`` ("nnodes > 1 can only be set when
distributed executor backend is mp, uni or external_launcher"), so the rank-0
process dies seconds after launch unless the image's entrypoint consumes those
options and starts the Ray cluster itself.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
NVIDIA_405B = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-dual"


def ray_vllm_recipes() -> list[tuple[str, dict]]:
    found = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        topology = recipe["topology"]
        if (
            recipe["runtime"]["engine"] == "vllm"
            and topology.get("node_count", 1) > 1
            and topology["parallelism"]["backend"] == "ray"
        ):
            found.append((path.stem, recipe))
    return found


def context_files(recipe: dict) -> list[Path]:
    context = ROOT / recipe["execution"]["build"]["context"]["path"]
    return [path for path in context.rglob("*") if path.is_file()]


class RayVllmLauncherTest(unittest.TestCase):
    def test_the_class_is_not_empty(self) -> None:
        self.assertIn(NVIDIA_405B, {slug for slug, _ in ray_vllm_recipes()})

    def test_every_ray_recipe_ships_a_launcher_that_consumes_the_placement_options(
        self,
    ) -> None:
        for slug, recipe in ray_vllm_recipes():
            with self.subTest(recipe=slug):
                launchers = [
                    path
                    for path in context_files(recipe)
                    if path.name != "Dockerfile"
                    and (text := path.read_text(errors="ignore"))
                    and "nnodes" in text
                    and "ray" in text
                    and "start" in text
                ]
                self.assertTrue(
                    launchers,
                    f"{slug} declares the Ray backend but ships no launcher that "
                    "consumes --nnodes and starts the Ray cluster; vLLM exits at "
                    "configuration time with --nnodes and the Ray executor",
                )

    def test_the_405b_context_fits_the_bounded_chat_check(self) -> None:
        recipe = json.loads((ROOT / "recipes" / f"{NVIDIA_405B}.json").read_text())
        context = recipe["settings"]["context_tokens"]["value"]
        arguments = {item["name"]: item for item in recipe["runtime"]["arguments"]}
        check = next(
            item
            for item in recipe["validation"]["serving"]["checks"]
            if item["name"] == "bounded-chat"
        )
        # The check asks for up to max_tokens completion tokens on top of the
        # chat-templated prompt (the playbook's 64-token context cannot hold both).
        self.assertGreater(context, check["request"]["body"]["max_tokens"] + 32)
        self.assertGreaterEqual(arguments["max-num-batched-tokens"]["value"], context)


class Llama405bWrapperTest(unittest.TestCase):
    wrapper = ROOT / "adapters/nvidia" / NVIDIA_405B / "vllm-wrapper.py"

    def run_wrapper(self, arguments: list[str], environment: dict[str, str]):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            log = bin_dir / "calls.log"
            for name in ("ray", "vllm"):
                stub = bin_dir / name
                stub.write_text(f'#!/bin/sh\necho "{name} $@" >> "{log}"\n')
                stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            env = {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                **environment,
            }
            result = subprocess.run(
                [sys.executable, str(self.wrapper), *arguments],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            calls = log.read_text().splitlines() if log.exists() else []
            return result, calls

    placement: ClassVar[dict[str, str]] = {
        "VONK_LOCAL_ADDR": "10.0.0.2",
        "VONK_MASTER_ADDR": "10.0.0.1",
        "VONK_MASTER_PORT": "6379",
    }

    def test_a_worker_joins_the_head_instead_of_serving(self) -> None:
        result, calls = self.run_wrapper(
            [
                "serve",
                "/models",
                "--distributed-executor-backend",
                "ray",
                "--nnodes",
                "2",
                "--node-rank",
                "1",
                "--headless",
            ],
            self.placement,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("ray start --address 10.0.0.1:6379"))
        self.assertTrue(all(not call.startswith("vllm") for call in calls))

    def test_a_launch_without_the_placement_is_refused(self) -> None:
        result, calls = self.run_wrapper(
            [
                "serve",
                "/models",
                "--distributed-executor-backend",
                "ray",
                "--nnodes",
                "2",
                "--node-rank",
                "0",
            ],
            {},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_the_wrapper_is_executable_and_shipped_as_the_entrypoint(self) -> None:
        dockerfile = (self.wrapper.parent / "Dockerfile").read_text()
        self.assertIn(
            "COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm", dockerfile
        )
        self.assertTrue(shutil.which("python3"))


if __name__ == "__main__":
    unittest.main()
