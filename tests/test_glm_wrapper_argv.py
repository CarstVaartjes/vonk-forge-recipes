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

RECIPES = {
    "glm-5-2-aqlm-vllm-triple": "adapters/glm/mia-triple/vllm-wrapper.py",
    "glm-5-2-quanttrio-vllm-four": "adapters/glm/eugr-four/vllm-wrapper.py",
    "glm-5-3-flash-exl3-dflash2-vllm-dual": "adapters/glm/mia-exl3-dflash2-dual/vllm-wrapper.py",
    "glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual": "adapters/glm/tonyd2wild-dflash2-dual/vllm-wrapper.py",
    "glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual": "adapters/glm/drowzeys-glm53-1m/vllm-wrapper.py",
    "glm-5-3-flash-nvfp4-vllm-dual": "adapters/glm/glm53-sm121/vllm-wrapper.py",
    "glm-5-3-flash-nvfp4-vllm-four": "adapters/glm/tonyd2wild-glm53-tp4-current/vllm-wrapper.py",
}


def load(slug: str) -> dict:
    return json.loads((ROOT / "recipes" / f"{slug}.json").read_text())


def compile_authored_argv(recipe: dict) -> list[str]:
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
    def test_every_glm_wrapper_executes_authored_argv_without_rewriting_engine_options(self) -> None:
        for slug, wrapper in RECIPES.items():
            with self.subTest(slug=slug):
                recipe = load(slug)
                if slug == "glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual":
                    authored_names = {item["name"]: item for item in recipe["runtime"]["arguments"]}
                    self.assertEqual(authored_names["attention-backend"]["value"], "B12X_MLA_SPARSE")
                authored = compile_authored_argv(recipe)
                authored_engine = authored[:]
                # These are Controller placement inputs, not engine argv.
                placement_start = authored_engine.index("--nnodes")
                authored_engine = authored_engine[:placement_start] + authored_engine[placement_start + 6 :]
                captured: list[tuple[str, ...]] = []

                def fake_execv(path: str, argv: tuple[str, ...]) -> None:
                    captured.append(argv)
                    raise StopExec

                def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                    return subprocess.CompletedProcess(args, 0, stdout=f"VONK_ALIVE={recipe['topology']['node_count']}\n")

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
                    if path == Path("/models/config.json"):
                        return '{"text_config":{"index_topk":2048}}'
                    return ""

                with patch.object(sys, "argv", [wrapper, *authored]), patch.dict(
                    os.environ, env, clear=False
                ), patch("os.execv", side_effect=fake_execv), patch(
                    "os.path.isfile", return_value=True
                ), patch("pathlib.Path.is_file", return_value=True), patch(
                    "os.access", return_value=True
                ), patch("subprocess.run", side_effect=fake_run), patch.object(
                    Path, "rglob", return_value=[]
                ), patch.object(Path, "mkdir"), patch.object(
                    Path, "write_text"
                ), patch.object(Path, "replace"), patch.object(
                    Path, "read_text", new=fake_read_text
                ):
                    with self.assertRaises(StopExec):
                        runpy.run_path(str(ROOT / wrapper))

                self.assertEqual(len(captured), 1)
                final = list(captured[0][1:])
                if slug == "glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual":
                    model_index = authored_engine.index("/models")
                    expected = authored_engine.copy()
                    expected[model_index] = "/outputs/glm53-index-topk-2044"
                else:
                    expected = authored_engine
                self.assertEqual(final[: len(expected)], expected)
                if recipe["topology"]["parallelism"]["backend"] == "mp":
                    self.assertEqual(
                        final[-4:],
                        ["--master-addr", "10.0.0.2", "--master-port", "29500"],
                    )
                self.assertEqual(final.count("--opaque-engine-option"), 2)
                self.assertEqual(final[final.index("--opaque-engine-option") + 1], "")
                last_opaque = max(index for index, value in enumerate(final) if value == "--opaque-engine-option")
                self.assertEqual(final[last_opaque + 1], "second")
                utilization = [
                    index for index, value in enumerate(final) if value == "--gpu-memory-utilization"
                ]
                self.assertGreaterEqual(len(utilization), 2)
                self.assertEqual(final[utilization[-1] + 1], "0.71")
                speculative = [
                    index for index, value in enumerate(final) if value == "--speculative-config"
                ]
                self.assertEqual(final[speculative[-1] + 1], '{"user_override":true}')


if __name__ == "__main__":
    unittest.main()
