"""Exercise managed configuration and native placement at new launcher boundaries."""

import importlib.util
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["c2", "c4", "c2-vit-off"])
def test_flash_next_profile_receives_platform_endpoint_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    import argparse

    # The pinned upstream parser accepts only profile, sources and print. The
    # platform also appends endpoint and memory arguments to every SGLang load.
    entrypoint = types.SimpleNamespace()
    entrypoint.MODEL_PATH = "/model"
    entrypoint.SERVE_HOST = "0.0.0.0"
    entrypoint.SERVE_PORT = 30000

    def build_argv(profile, sources):
        return [
            "sglang",
            "serve",
            "--model-path",
            entrypoint.MODEL_PATH,
            "--host",
            entrypoint.SERVE_HOST,
            "--port",
            str(entrypoint.SERVE_PORT),
            "--mem-fraction-static",
            "0.9",
            "--speculative-num-steps",
            "3",
        ]

    entrypoint.build_argv = build_argv
    captured = []

    def main(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--profile", required=True)
        parser.add_argument("--sources", required=True)
        args = parser.parse_args(argv)
        assert args.profile == profile
        assert args.sources == "sources.json"
        captured.extend(entrypoint.build_argv(args.profile, args.sources))
        return 0

    entrypoint.main = main
    runtime = types.ModuleType("runtime")
    runtime.__dict__["entrypoint"] = entrypoint
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sglang-serve",
            "--profile",
            profile,
            "--sources",
            "sources.json",
            "--mem-fraction-static",
            "0.75",
            "--host",
            "127.0.0.1",
            "--port",
            "31000",
        ],
    )
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "model-fp8-mtp-ple.safetensors",
    ):
        (tmp_path / name).touch()
    source = (
        ROOT / "adapters/qwen/r0b0tlab-flash-next-sglang-single/flash-next-serve"
    ).read_text()
    launcher = tmp_path / "launcher"
    launcher.write_text(
        source.replace('Path("/models")', f"Path({str(tmp_path)!r})").replace(
            'Path("/outputs/cache")', f"Path({str(tmp_path / 'cache')!r})"
        )
    )
    with pytest.raises(SystemExit) as ended:
        runpy.run_path(str(launcher), run_name="__main__")
    assert ended.value.code == 0
    for flag, value in (
        ("--model-path", str(tmp_path)),
        ("--host", "127.0.0.1"),
        ("--port", "31000"),
        ("--mem-fraction-static", "0.75"),
        ("--speculative-num-steps", "3"),
    ):
        assert captured.count(flag) == 1
        assert captured[captured.index(flag) + 1] == value


@pytest.mark.parametrize("memory", ["nan", "inf", "0", "1", "-0.5"])
def test_flash_next_rejects_invalid_platform_memory(memory: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "adapters/qwen/r0b0tlab-flash-next-sglang-single/flash-next-serve"
            ),
            "--profile",
            "unused",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--mem-fraction-static",
            memory,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "mem-fraction-static must be finite and between 0 and 1" in result.stderr


@pytest.mark.parametrize(
    "profile,context,vision,cache",
    [("vision", 4096, True, "FP16"), ("long", 262144, False, "Q4")],
)
def test_minicpm_profile_writes_cache_without_mutating_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    context: int,
    vision: bool,
    cache: str,
) -> None:
    adapter = ROOT / "adapters/minicpm/vcruz305-exl3-single"
    spec = importlib.util.spec_from_file_location("common", adapter / "tools/common.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "common", module)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["launcher", "--profile", profile, "--max-seq-len", str(context)]
    )
    monkeypatch.setattr(os, "chdir", lambda path: None)
    calls = []
    monkeypatch.setattr(
        os, "execv", lambda executable, argv: calls.append((executable, argv))
    )
    runpy.run_path(str(adapter / "exllamav3-serve"), run_name="__main__")
    config = tmp_path / "minicpm-tabby/config.yml"
    lines = config.read_text().splitlines()
    assert f"  vision: {json.dumps(vision)}" in lines
    assert f"  max_seq_len: {context}" in lines
    assert f'  cache_mode: "{cache}"' in lines
    assert (tmp_path / "minicpm-tabby/models/MiniCPM-V-4.7").readlink() == Path(
        "/models/target"
    )
    assert calls[0][1][-2:] == ["--config", str(config)]


@pytest.mark.parametrize("valid", [False, True])
def test_fp4_dual_binds_controller_placement(tmp_path: Path, valid: bool) -> None:
    package = tmp_path / "vllm/entrypoints/cli"
    package.mkdir(parents=True)
    (package / "main.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(tmp_path),
        VONK_LOCAL_ADDR="192.0.2.2",
        VONK_MASTER_ADDR="192.0.2.1" if valid else "bad",
        VONK_MASTER_PORT="29500",
    )
    recipe = json.loads(
        (
            ROOT / "recipes/qwen3-8-flash-next-fp4-local-nvidia-vllm-dual.json"
        ).read_text()
    )
    prefix = recipe["runtime"]["entrypoint"][1:]
    assert prefix == ["serve", "/models"]
    argv = [
        *["/models/target" if value == "/models" else value for value in prefix],
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--headless",
    ]
    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "adapters/nvidia/qwen3-8-flash-next-fp4-local-nvidia-vllm-dual/vllm-wrapper"
            ),
            *argv,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if valid:
        assert result.returncode == 0
        assert json.loads(result.stdout) == [
            *argv,
            "--load-format",
            "safetensors",
            "--master-addr",
            "192.0.2.1",
            "--master-port",
            "29500",
        ]
    else:
        assert result.returncode != 0
        assert not result.stdout
