"""Exercise launcher argument preservation and fail-closed rendezvous resolution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [
    ("qwen/qwen3-8-27b-taussoe-tensorfold-single", "/models/target", "tensorfold"),
    ("qwen/flash-next-taussoe-tensorfold-single", "/models", "tensorfold"),
    (
        "qwen/mia-flash-next-tensorfold-dual-nvfp4",
        "/models/target",
        "tensorfold-native",
    ),
    (
        "qwen/mia-flash-next-tensorfold-dual-int4-autoround",
        "/models/target",
        "tensorfold-native",
    ),
    ("nemotron/mia-bolt-tensorfold-single", "/models/target", "tensorfold-native"),
    ("deepseek/sfxnz-v41-tensorfold-dual", "/models/target", "tensorfold"),
    ("glm/mia-full-tensorfold-triple", "/models/target", "tensorfold"),
]


@pytest.mark.parametrize(("adapter", "model_root", "engine"), LAUNCHERS)
@pytest.mark.parametrize("distributed", [False, True])
def test_launcher_preserves_arguments_and_resolves_placement(
    tmp_path: Path, adapter: str, model_root: str, engine: str, distributed: bool
) -> None:
    executable = tmp_path / engine
    executable.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    for name in ("HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        env[name] = str(tmp_path / name)
    args = ["--name", "model with spaces", "--context", "262144"]
    expected = list(args)
    if distributed:
        env.update(VONK_MASTER_ADDR="192.0.2.10", VONK_MASTER_PORT="29500")
        args += [
            "--tp",
            "2",
            "--rank",
            "1",
            "--master",
            "VONK_MASTER_ADDR",
            "--master-port",
            "VONK_MASTER_PORT",
        ]
        expected += [
            "--tp",
            "2",
            "--rank",
            "1",
            "--master",
            "192.0.2.10",
            "--master-port",
            "29500",
        ]
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "adapters" / adapter / "tensorfold-serve"),
            *args,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == ["serve", model_root, *expected]
    assert (tmp_path / "HOME/.cache/tensorfold").is_dir()


@pytest.mark.parametrize(
    "address,port",
    [("bad-address", "29500"), ("192.0.2.10", "80"), ("192.0.2.10", "65536")],
)
def test_invalid_placement_never_executes_engine(
    tmp_path: Path, address: str, port: str
) -> None:
    marker = tmp_path / "executed"
    executable = tmp_path / "tensorfold"
    executable.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    executable.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        VONK_MASTER_ADDR=address,
        VONK_MASTER_PORT=port,
    )
    for name in ("HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        env[name] = str(tmp_path / name)
    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "adapters/qwen/qwen3-8-27b-taussoe-tensorfold-single/tensorfold-serve"
            ),
            "--master",
            "VONK_MASTER_ADDR",
            "--master-port",
            "VONK_MASTER_PORT",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "needs a valid" in result.stderr
    assert not marker.exists()
