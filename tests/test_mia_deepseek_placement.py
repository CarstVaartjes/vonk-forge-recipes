"""The native Engram adapter must account for one local rank, preserving authorized fabric."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "mia_deepseek_placement",
    Path(__file__).resolve().parents[1]
    / "adapters/deepseek/mia-1017-sglang/distributed.py",
)
assert SPEC and SPEC.loader
PLACEMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLACEMENT)


@pytest.mark.parametrize("nodes", [3, 4])
def test_placement_binds_rank_accounting_without_replacing_fabric(nodes):
    environment = {
        "VONK_MASTER_ADDR": "192.0.2.1",
        "VONK_LOCAL_ADDR": "192.0.2.2",
        "VONK_MASTER_PORT": "29500",
        "NCCL_SOCKET_IFNAME": "=fabric0",
        "NCCL_IB_HCA": "=rocep1s0:1,rocep2s0:1",
        "NCCL_IB_GID_INDEX": "3",
        "TP_SOCKET_IFNAME": "fabric0",
        "GLOO_SOCKET_IFNAME": "fabric0",
    }
    args = [
        "--nnodes",
        str(nodes),
        "--node-rank",
        str(nodes - 1),
        "--dist-init-addr",
        "VONK_MASTER_ADDR:VONK_MASTER_PORT",
        "--arbitrary-engine-option",
        "value",
    ]
    command, result = PLACEMENT.bind(args, environment)
    assert command[-2:] == args[-2:]
    assert command[5] == "192.0.2.1:29500"
    assert result["NNODES"] == str(nodes)
    assert result["NODE_RANK"] == str(nodes - 1)
    assert {key: result[key] for key in environment} == environment
    assert (result.get("DSV41_LAUNCHER") == "tp4") == (nodes == 4)
    with pytest.raises(ValueError, match="fabric"):
        PLACEMENT.bind(args, {**environment, "NCCL_IB_HCA": ""})
    with pytest.raises(ValueError, match="port"):
        PLACEMENT.bind(args, {**environment, "VONK_MASTER_PORT": "0"})
    invalid = list(args)
    invalid[3] = str(nodes)
    with pytest.raises(ValueError, match="topology"):
        PLACEMENT.bind(invalid, environment)
