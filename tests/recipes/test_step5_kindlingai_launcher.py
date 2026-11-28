"""KindlingAI's Step-5 Preview launcher binds Controller placement and the FP8 KV overlay."""

import os
import subprocess
from pathlib import Path

import pytest
from repo_root import ROOT

LAUNCHER = ROOT / "adapters/stepfun/kindlingai-vllm-four/launcher"
ARGS = [
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--hf-overrides",
    "{}",
    "--distributed-executor-backend",
    "mp",
    "--nnodes",
    "4",
]
FABRIC = dict.fromkeys(
    (
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "TP_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    ),
    "x",
)


@pytest.fixture
def run(tmp_path):
    """The launcher with /models, the overlay and vllm moved under tmp_path; returns (done, argv, env)."""
    models = tmp_path / "models"
    models.mkdir()
    for name in ("config.json", "model.safetensors.index.json"):
        (models / name).write_text("{}")
    (tmp_path / "overlay/step5_fp8kv").mkdir(parents=True)
    vllm = tmp_path / "vllm"
    vllm.write_text('#!/bin/bash\nenv > "$DUMP"\nprintf "%s\\n" "$@" > "$DUMP.argv"\n')
    vllm.chmod(0o755)
    text = LAUNCHER.read_text()
    text = text.replace('Path("/models")', f'Path("{models}")')
    text = text.replace(
        'Path("/opt/step5-fp8kv/overlay")', f'Path("{tmp_path}/overlay")'
    )
    text = text.replace("/usr/local/bin/vllm", str(vllm))
    script = tmp_path / "launcher"
    script.write_text(text)

    def go(rank, env, extra=()):
        argv = [
            *ARGS,
            "--node-rank",
            str(rank),
            *(["--headless"] if rank else []),
            *extra,
        ]
        done = subprocess.run(
            ["python3", str(script), *argv],
            env={
                "PATH": os.environ["PATH"],
                "DUMP": str(tmp_path / "dump"),
                "VONK_LOCAL_ADDR": f"10.9.0.{rank + 1}",
                "VONK_MASTER_ADDR": "10.9.0.1",
                "VONK_MASTER_PORT": "29555",
                **FABRIC,
                **env,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        dump = tmp_path / "dump"
        seen = (
            dict(line.partition("=")[::2] for line in dump.read_text().splitlines())
            if dump.exists()
            else {}
        )
        args = Path(f"{dump}.argv").read_text().splitlines() if dump.exists() else []
        return done, args, seen

    return go, tmp_path


def test_fp8_kv_adds_the_overlay_and_block_arguments(run):
    go, root = run
    done, args, env = go(2, {"STEP5_FP8KV": "1", "FP8KV_BLOCKS": "30000"})
    assert done.returncode == 0, done.stderr
    assert args[0] == "serve"
    assert args[-8:] == [
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "30000",
        "--master-addr",
        "10.9.0.1",
        "--master-port",
        "29555",
    ]
    assert env["PYTHONPATH"] == f"{root}/overlay"
    assert (env["VLLM_HOST_IP"], env["MASTER_ADDR"]) == ("10.9.0.3", "10.9.0.1")


def test_bf16_kv_runs_without_the_overlay(run):
    go, _ = run
    done, args, env = go(0, {"STEP5_FP8KV": "0"})
    assert done.returncode == 0, done.stderr
    assert "--block-size" not in args
    assert "PYTHONPATH" not in env
    assert env["STEP5_FP8KV"] == "0"


@pytest.mark.parametrize(
    "env",
    [{"STEP5_FP8KV": "1"}, {"STEP5_FP8KV": "1", "FP8KV_BLOCKS": "0"}],
)
def test_fp8_kv_needs_a_block_count(run, env):
    go, _ = run
    done, args, _ = go(0, env)
    assert done.returncode != 0
    assert not args


def test_incomplete_rendezvous_is_refused(run):
    go, _ = run
    done, args, _ = go(1, {"VONK_MASTER_PORT": ""})
    assert done.returncode != 0
    assert not args
