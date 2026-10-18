"""Exercise managed configuration and native placement at new launcher boundaries."""

import importlib.util
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
            "--master-addr",
            "192.0.2.1",
            "--master-port",
            "29500",
        ]
    else:
        assert result.returncode != 0
        assert not result.stdout
