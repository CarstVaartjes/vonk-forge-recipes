"""The Controller launcher binds one rank of sfxnz's TensorFold 1.2.5 GLM pair to its environment."""

import os
import runpy
import sys

import pytest
from repo_root import ROOT

# The sfxnz pair, the Sage pair and the XSFP4 recipes run the same stock-release launch; every test below runs against
# every launcher. The XSFP4 image installs the engine at /usr/local/bin.
ENGINES = {
    ROOT
    / "adapters/glm/sfxnz-tensorfold-native-dual/tensorfold-serve": "/opt/tensorfold/bin/tensorfold-native",
    ROOT
    / "adapters/glm/bizuayeu-sage-tensorfold-dual/tensorfold-serve": "/opt/tensorfold/bin/tensorfold-native",
    ROOT
    / "adapters/glm/bizuayeu-xsfp4-tensorfold-native/tensorfold-serve": "/usr/local/bin/tensorfold-native",
}
LAUNCHERS = list(ENGINES)
# Sage passes the platform's rank flags straight to the engine (TensorFold 1.3.5's documented form); the others still use
# the environment aliases.
NATIVE_FLAGS = {ROOT / "adapters/glm/bizuayeu-sage-tensorfold-dual/tensorfold-serve"}
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


@pytest.fixture(params=LAUNCHERS, ids=lambda path: path.parent.name)
def launcher(request):
    return request.param


def launch(launcher, monkeypatch, tmp_path, arguments, **environ):
    for name in list(os.environ):
        if name.startswith(("TF_", "NCCL_", "VONK_")):
            monkeypatch.delenv(name)
    for name, value in {**ENVIRON, "TMPDIR": str(tmp_path / "tmp"), **environ}.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, argv)))
    monkeypatch.setattr(sys, "argv", ["tensorfold-serve", *arguments])
    runpy.run_path(str(launcher), run_name="__main__")
    return calls


def test_rank_binds_the_pair_for_the_engine(launcher, monkeypatch, tmp_path):
    (call,) = launch(launcher, monkeypatch, tmp_path, [*SERVE, *PLATFORM])
    engine = ENGINES[launcher]
    if launcher in NATIVE_FLAGS:
        resolved = [*PLATFORM[:4], "--master", "192.0.2.1", "--master-port", "29500"]
        assert call == (engine, [engine, "serve", "/models/target", *SERVE, *resolved])
        assert not {"TF_GLM_RANK", "TF_GLM_WORLD", "NCCL_COMM_ID"} & os.environ.keys()
    else:
        assert call == (engine, [engine, "serve", "/models/target", *SERVE])
        assert os.environ["TF_GLM_RANK"] == "1"
        assert os.environ["TF_GLM_WORLD"] == "2"
        assert os.environ["NCCL_COMM_ID"] == "192.0.2.1:29500"
    assert os.environ["TF_ROCE_HCA"] == "rocep1s0f1,roceP2p1s0f1"
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
def test_missing_or_malformed_placement_is_refused(
    launcher, monkeypatch, tmp_path, environ
):
    with pytest.raises(SystemExit, match="tensorfold-serve"):
        launch(launcher, monkeypatch, tmp_path, [*SERVE, *PLATFORM], **environ)


def test_a_rank_outside_the_world_is_refused(launcher, monkeypatch, tmp_path):
    arguments = [*SERVE, *PLATFORM]
    arguments[arguments.index("--rank") + 1] = "2"
    with pytest.raises(SystemExit, match="--rank"):
        launch(launcher, monkeypatch, tmp_path, arguments)
