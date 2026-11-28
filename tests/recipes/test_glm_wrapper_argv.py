from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from repo_root import ROOT

RECIPES = {
    "glm-5-2-quanttrio-vllm-four": "adapters/glm/eugr-four/vllm-wrapper.py",
    "glm-5-3-flash-exl3-dflash2-vllm-mia-four": "adapters/glm/mia-exl3-dflash2-four/vllm-wrapper.py",
    "glm-5-3-flash-exl3-dflash2-vllm-mia-triple": "adapters/glm/mia-exl3-dflash2-triple/vllm-wrapper.py",
    "glm-5-3-flash-nvfp4-dflash2-kindlingai-vllm-dual": "adapters/glm/kindlingai-glm53f-nvfp4-vllm-dual/vllm-wrapper.py",
    "glm-5-3-flash-exlr8-kindlingai-vllm-dual": "adapters/glm/kindlingai-glm53f-exlr8-vllm-dual/vllm-wrapper.py",
    "glm-5-3-flash-nvfp4-vllm-dual": "adapters/glm/glm53-sm121/vllm-wrapper.py",
    "glm-5-3-flash-nvfp4-vllm-four": "adapters/glm/tonyd2wild-glm53-tp4-current/vllm-wrapper.py",
    "minimax-m3-1-flash-tonyd2wild-vllm-four": "adapters/minimax/tonyd2wild-m31-flash-tp4/vllm-wrapper.py",
    "minimax-m3-1-flash-exl3-sage-2-15bpw-tonyd2wild-vllm-dual": "adapters/minimax/tonyd2wild-m31-flash-exl3-tp2/vllm-wrapper.py",
}


def load(slug: str) -> dict[str, Any]:
    return json.loads((ROOT / "recipes" / f"{slug}.json").read_text())


def compile_authored_argv(recipe: dict[str, Any]) -> list[str]:
    """Render the recipe's scalar runtime arguments as the vLLM argv contract."""
    settings = recipe["settings"]
    arguments = list(recipe["runtime"]["entrypoint"][1:])
    for item in recipe["runtime"]["arguments"]:
        value = item.get("value")
        if value is None:
            value = settings[item["setting"]]["value"]
        if type(value) is bool:
            if value:
                arguments.append(f"--{item['name']}")
        else:
            arguments.extend((f"--{item['name']}", str(value)))
    arguments.extend(
        (
            "--opaque-engine-option",
            "",
            "--opaque-engine-option",
            "second",
            "--gpu-memory-utilization",
            "0.69",
            "--gpu-memory-utilization",
            "0.71",
            "--speculative-config",
            '{"user_override":true}',
            "--nnodes",
            str(recipe["topology"]["node_count"]),
            "--node-rank",
            "0",
            "--distributed-executor-backend",
            recipe["topology"]["parallelism"]["backend"],
        )
    )
    return arguments


class StopExec(RuntimeError):
    pass


class GlmWrapperArgvTests(unittest.TestCase):
    def test_every_glm_wrapper_executes_authored_argv_without_rewriting_engine_options(
        self,
    ) -> None:
        for slug, wrapper in RECIPES.items():
            with self.subTest(slug=slug):
                recipe = load(slug)
                authored = compile_authored_argv(recipe)
                authored_engine = authored[:]
                # These are Controller placement inputs, not engine argv.
                placement_start = authored_engine.index("--nnodes")
                authored_engine = (
                    authored_engine[:placement_start]
                    + authored_engine[placement_start + 6 :]
                )
                captured: list[tuple[str, ...]] = []

                def fake_execv(
                    path: str,
                    argv: tuple[str, ...],
                    captured: list[tuple[str, ...]] = captured,
                ) -> None:
                    captured.append(argv)
                    raise StopExec

                def fake_run(
                    *args: object,
                    _recipe: dict[str, Any] = recipe,
                    **kwargs: object,
                ) -> subprocess.CompletedProcess[str]:
                    return subprocess.CompletedProcess(
                        args,  # pyright: ignore[reportArgumentType]
                        0,
                        stdout=(
                            f"VONK_ALIVE={_recipe['topology']['node_count']}\n"
                            f"Resources: 0.0/{_recipe['topology']['node_count']}.0 GPU\n"
                        ),
                    )

                env = {
                    "VONK_LOCAL_ADDR": "10.0.0.1",
                    "VONK_MASTER_ADDR": "10.0.0.2",
                    "VONK_MASTER_PORT": "29500",
                    "NCCL_SOCKET_IFNAME": "eth0",
                    "NCCL_IB_HCA": "roce0",
                    "NCCL_IB_GID_INDEX": "3",
                    "TP_SOCKET_IFNAME": "eth0",
                    "GLOO_SOCKET_IFNAME": "eth0",
                }

                def fake_read_text(path: Path, *args: object, **kwargs: object) -> str:
                    return ""

                with (
                    patch.object(sys, "argv", [wrapper, *authored]),
                    patch.dict(os.environ, env, clear=False),
                    patch("os.execv", side_effect=fake_execv),
                    patch("os.path.isfile", return_value=True),
                    patch("pathlib.Path.is_file", return_value=True),
                    patch("os.access", return_value=True),
                    patch("subprocess.run", side_effect=fake_run),
                    patch("subprocess.Popen"),
                    patch.object(Path, "rglob", return_value=[]),
                    patch.object(Path, "mkdir"),
                    patch.object(Path, "write_text"),
                    patch.object(Path, "replace"),
                    patch.object(Path, "read_text", new=fake_read_text),
                    pytest.raises(StopExec),
                ):
                    runpy.run_path(str(ROOT / wrapper))

                assert len(captured) == 1
                final = list(captured[0][1:])
                expected = authored_engine
                assert final[: len(expected)] == expected
                if recipe["topology"]["parallelism"]["backend"] == "mp":
                    assert final[-4:] == [
                        "--master-addr",
                        "10.0.0.2",
                        "--master-port",
                        "29500",
                    ]
                assert final.count("--opaque-engine-option") == 2
                assert final[final.index("--opaque-engine-option") + 1] == ""
                last_opaque = max(
                    index
                    for index, value in enumerate(final)
                    if value == "--opaque-engine-option"
                )
                assert final[last_opaque + 1] == "second"
                utilization = [
                    index
                    for index, value in enumerate(final)
                    if value == "--gpu-memory-utilization"
                ]
                assert len(utilization) >= 2
                assert final[utilization[-1] + 1] == "0.71"
                speculative = [
                    index
                    for index, value in enumerate(final)
                    if value == "--speculative-config"
                ]
                assert final[speculative[-1] + 1] == '{"user_override":true}'


MENTATS = (
    "glm-5-3-flash-nvfp4-dflash2-kindlingai-vllm-dual",
    "glm-5-3-flash-exlr8-kindlingai-vllm-dual",
)


def run_mentat_launcher(
    slug: str,
    rank: int,
    extra_env: dict[str, str] | None = None,
    gpus: int = 2,
    daemon_exit: int | None = None,
    expect: type[BaseException] = StopExec,
) -> tuple[list[Any], list[Any], list[Any], list[str]]:
    """What the mentat launcher starts for one rank: Popen, run and exec calls, and the directories made."""
    wrapper = RECIPES[slug]
    recipe = load(slug)
    arguments = compile_authored_argv(recipe)
    arguments[arguments.index("--node-rank") + 1] = str(rank)
    if rank:
        arguments.append("--headless")
    env = {
        "VONK_LOCAL_ADDR": f"10.0.0.{rank + 1}",
        "VONK_MASTER_ADDR": "10.0.0.1",
        "VONK_MASTER_PORT": "29500",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_HCA": "roce0",
        "NCCL_IB_GID_INDEX": "3",
        "TP_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
        **(extra_env or {}),
    }
    popen: list[Any] = []
    run: list[Any] = []
    execs: list[Any] = []
    made: list[str] = []

    def fake_popen(command: list[str]) -> Any:
        popen.append(command)
        return SimpleNamespace(poll=lambda: daemon_exit)

    def fake_execv(path: str, argv: tuple[str, ...]) -> None:
        execs.append((path, argv))
        raise StopExec

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        run.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=f"Resources: 0.0/{gpus}.0 GPU (0.0 reserved)\n"
        )

    with (
        patch.object(sys, "argv", [wrapper, *arguments]),
        patch.dict(os.environ, env, clear=False),
        patch("os.execv", side_effect=fake_execv),
        patch("pathlib.Path.is_file", return_value=True),
        patch("os.access", return_value=True),
        patch("subprocess.run", side_effect=fake_run),
        patch("subprocess.Popen", side_effect=fake_popen),
        patch.object(Path, "rglob", return_value=[]),
        patch.object(
            Path,
            "mkdir",
            autospec=True,
            side_effect=lambda path, **_: made.append(str(path)),
        ),
        patch.object(Path, "read_text", new=lambda *args, **kwargs: ""),
        patch("time.sleep"),
        pytest.raises(expect),
    ):
        runpy.run_path(str(ROOT / wrapper))
    return popen, run, execs, made


class MentatLauncherTests(unittest.TestCase):
    def test_rank_zero_hosts_the_daemon_on_the_rendezvous_port_then_serves(
        self,
    ) -> None:
        for slug in MENTATS:
            self.check_rank_zero(slug)

    def check_rank_zero(self, slug: str) -> None:
        popen, run, execs, _ = run_mentat_launcher(slug, 0)
        assert popen == [
            [
                "/usr/local/bin/mentatd",
                "daemon",
                "--port",
                "29500",
                "--node-ip",
                "10.0.0.1",
            ]
        ]
        assert run[0][:3] == ["/usr/local/bin/ray", "start", "--head"]
        assert "--address=127.0.0.1:29500" in run[0]
        assert run[1][:3] == ["/usr/local/bin/ray", "status", "--address"]
        path, argv = execs[0]
        assert path == "/usr/local/bin/vllm"
        for placement in ("--nnodes", "--node-rank", "--headless"):
            assert placement not in argv
        assert "--tensor-parallel-size" in argv

    def test_the_control_file_variables_are_never_made_into_directories(self) -> None:
        names = {
            "VLLM_ADAPTIVE_K_CONTROL": "/outputs/cache/adaptive-k",
            "VLLM_DRAFT_TRUNC_CONTROL": "/outputs/cache/draft-trunc",
            "VLLM_WEIGHT_SNAPSHOT_DIR": "/outputs/cache/weight-snapshots",
        }
        for slug in MENTATS:
            *_, made = run_mentat_launcher(slug, 0, extra_env=names)
            assert "/outputs/cache/weight-snapshots" in made
            assert not {
                names["VLLM_ADAPTIVE_K_CONTROL"],
                names["VLLM_DRAFT_TRUNC_CONTROL"],
            } & set(made)

    def test_a_daemon_that_exits_while_the_group_forms_stops_the_launch(self) -> None:
        # The helper requires the SystemExit; vLLM is never exec'd before it.
        for slug in MENTATS:
            _, _, execs, _ = run_mentat_launcher(
                slug, 0, gpus=0, daemon_exit=1, expect=SystemExit
            )
            assert not execs

    def test_a_worker_joins_the_rank_zero_daemon_without_starting_one(self) -> None:
        popen, run, execs, _ = run_mentat_launcher(MENTATS[0], 1)
        assert not popen
        assert not run
        assert execs == [
            (
                "/usr/local/bin/ray",
                (
                    "/usr/local/bin/ray",
                    "start",
                    "--block",
                    "--address=10.0.0.1:29500",
                ),
            )
        ]


if __name__ == "__main__":
    unittest.main()
