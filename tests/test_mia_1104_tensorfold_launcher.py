"""The Mia 1104 GLM dual wrapper turns the Controller's placement into the kit's per-rank environment, or refuses."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "adapters/glm/mia-1104-tensorfold-dual/tensorfold-serve"
)
FABRIC = {
    "VONK_MASTER_ADDR": "192.0.2.10",
    "VONK_MASTER_PORT": "29500",
    "NCCL_IB_HCA": "=rocep1s0f0:1,roceP2p1s0f0:1",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
}
PLACEMENT = [
    "--tp",
    "2",
    "--rank",
    "0",
    "--master",
    "VONK_MASTER_ADDR",
    "--master-port",
    "VONK_MASTER_PORT",
]


def seen(stdout: str) -> tuple[list[str], dict[str, str]]:
    argv = [line[2:] for line in stdout.splitlines() if line.startswith("A:")]
    env = dict(
        line[2:].split("=", 1) for line in stdout.splitlines() if line.startswith("E:")
    )
    return argv, env


def run(tmp_path: Path, args: list[str], **env: str):
    stub = tmp_path / "bin" / "tensorfold-native"
    stub.parent.mkdir(exist_ok=True)
    stub.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "for a in sys.argv[1:]: print('A:' + a)\n"
        "for k, v in os.environ.items(): print('E:' + k + '=' + v)\n"
    )
    stub.chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in FABRIC}
    base.update(
        PATH=f"{stub.parent}:{os.environ['PATH']}",
        HOME=str(tmp_path / "home"),
        TMPDIR=str(tmp_path / "tmp"),
    )
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        env={**base, **env},
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("rank", ["0", "1"])
def test_both_ranks_get_the_kits_placement(tmp_path: Path, rank: str) -> None:
    placement = list(PLACEMENT)
    placement[3] = rank
    result = run(tmp_path, ["--port", "8888", "--vision", *placement], **FABRIC)
    assert result.returncode == 0, result.stderr
    argv, env = seen(result.stdout)
    assert argv == [
        "serve",
        "/models/target",
        "--port",
        "8888",
        "--vision",
        "--chat-template",
        "/opt/vonk/templates/chat_template.jinja",
    ]
    assert env["TF_GLM_RANK"] == rank
    assert env["TF_GLM_WORLD"] == "2"
    assert env["NCCL_COMM_ID"] == "192.0.2.10:29500"
    assert env["TF_ROCE_HCA"] == "rocep1s0f0,roceP2p1s0f0"
    assert env["NCCL_IB_HCA"] == FABRIC["NCCL_IB_HCA"]
    assert (tmp_path / "home").is_dir() and (tmp_path / "tmp").is_dir()


def test_text_only_keeps_the_checkpoints_template(tmp_path: Path) -> None:
    result = run(tmp_path, ["--port", "8888", *PLACEMENT], **FABRIC)
    assert seen(result.stdout)[0] == [
        "serve",
        "/models/target",
        "--port",
        "8888",
    ]


@pytest.mark.parametrize(
    "missing",
    [
        "VONK_MASTER_ADDR",
        "VONK_MASTER_PORT",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_SOCKET_IFNAME",
    ],
)
def test_a_missing_variable_does_not_start_the_engine(
    tmp_path: Path, missing: str
) -> None:
    result = run(
        tmp_path, list(PLACEMENT), **{k: v for k, v in FABRIC.items() if k != missing}
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "tensorfold-serve:" in result.stderr


@pytest.mark.parametrize(
    "bad",
    [
        {"NCCL_IB_HCA": "=a:1,a:1"},
        {"NCCL_IB_HCA": "=a:1,b:1,c:1"},
        {"NCCL_IB_HCA": "=a:2"},
        {"NCCL_IB_GID_INDEX": "x"},
        {"VONK_MASTER_ADDR": "not-an-address"},
        {"VONK_MASTER_PORT": "80"},
    ],
)
def test_malformed_placement_does_not_start_the_engine(
    tmp_path: Path, bad: dict[str, str]
) -> None:
    result = run(tmp_path, list(PLACEMENT), **{**FABRIC, **bad})
    assert result.returncode != 0 and result.stdout == ""


def test_other_topologies_are_refused(tmp_path: Path) -> None:
    result = run(tmp_path, ["--tp", "3", *PLACEMENT[2:]], **FABRIC)
    assert result.returncode != 0 and result.stdout == ""
