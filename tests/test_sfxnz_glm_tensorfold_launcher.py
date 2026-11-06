"""The Controller launcher binds one rank of sfxnz's TensorFold 1.2.2 GLM pair to its environment."""

import os
import runpy
import sys
from pathlib import Path

import pytest

LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "adapters/glm/sfxnz-tensorfold-native-dual/tensorfold-serve"
)
SERVE = ["--host", "0.0.0.0", "--port", "8000", "--vision", "--parallel", "8"]
PLATFORM = [
    "--tp",
    "2",
    "--rank",
    "1",
    "--master",
    "VONK_MASTER_ADDR",
    "--master-port",
    "VONK_MASTER_PORT",
]
ENVIRON = {
    "HOME": "/outputs/cache/home",
    "VONK_MASTER_ADDR": "192.0.2.1",
    "VONK_MASTER_PORT": "29500",
    "NCCL_SOCKET_IFNAME": "enp1s0f1np1",
    "NCCL_IB_HCA": "=rocep1s0f1:1,roceP2p1s0f1:1",
}


def launch(monkeypatch, tmp_path, arguments, **environ):
    for name in list(os.environ):
        if name.startswith(("TF_", "NCCL_", "VONK_")):
            monkeypatch.delenv(name)
    for name, value in {**ENVIRON, "TMPDIR": str(tmp_path / "tmp"), **environ}.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, argv)))
    monkeypatch.setattr(sys, "argv", ["tensorfold-serve", *arguments])
    runpy.run_path(str(LAUNCHER), run_name="__main__")
    return calls


def test_rank_binds_the_pair_through_the_engines_environment(monkeypatch, tmp_path):
    (call,) = launch(monkeypatch, tmp_path, [*SERVE, *PLATFORM])
    assert call == (
        "/opt/tensorfold/bin/tensorfold-native",
        ["/opt/tensorfold/bin/tensorfold-native", "serve", "/models/target", *SERVE],
    )
    assert os.environ["TF_GLM_RANK"] == "1"
    assert os.environ["TF_GLM_WORLD"] == "2"
    assert os.environ["NCCL_COMM_ID"] == "192.0.2.1:29500"
    assert os.environ["TF_ROCE_HCA"] == "rocep1s0f1"
    assert os.environ["NCCL_IB_HCA"] == ENVIRON["NCCL_IB_HCA"]


@pytest.mark.parametrize(
    "environ",
    [
        {"VONK_MASTER_ADDR": "not-an-address"},
        {"VONK_MASTER_PORT": "80"},
        {"NCCL_IB_HCA": ""},
        {"NCCL_IB_HCA": "=rocep1s0f1:2"},
        {"NCCL_IB_HCA": "=rocep1s0f1:1,rocep1s0f1:1"},
        {"NCCL_SOCKET_IFNAME": ""},
    ],
)
def test_missing_or_malformed_placement_is_refused(monkeypatch, tmp_path, environ):
    with pytest.raises(SystemExit, match="tensorfold-serve"):
        launch(monkeypatch, tmp_path, [*SERVE, *PLATFORM], **environ)


def test_a_rank_outside_the_world_is_refused(monkeypatch, tmp_path):
    arguments = [*SERVE, *PLATFORM]
    arguments[arguments.index("--rank") + 1] = "2"
    with pytest.raises(SystemExit, match="--rank"):
        launch(monkeypatch, tmp_path, arguments)
