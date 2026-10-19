"""Native rank preparation and rendezvous must survive restart without changing inputs."""

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    params=[
        (2, "tonyd2wild-1017-tensorfold-tp2"),
        (4, "tonyd2wild-1017-tensorfold-tp4"),
        (2, "mia-1017-tensorfold-dual"),
    ]
)
def modules(request, monkeypatch):
    world, adapter = request.param
    folder = ROOT / "adapters/deepseek" / adapter
    loaded = []
    for name in ("prepare_cache", "launcher"):
        spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return world, *loaded


def test_rank_cache_reuses_complete_and_rebuilds_truncated(
    modules, tmp_path, monkeypatch
):
    world, cache, _ = modules
    expected = [["tensor", "uint8", [4], 0, 4]]
    body = json.dumps(expected).encode()
    name = f"rank0of{world}-{hashlib.sha256(body).hexdigest()[:20]}.bin"
    output = tmp_path / "rank-cache"
    calls = []

    def generate(command, check):
        assert check
        calls.append(command)
        (output / name).write_bytes(b"data" + body + struct.pack("<Q", 4) + b"TFDSRK01")

    monkeypatch.setattr(cache.subprocess, "check_output", lambda command: body)
    monkeypatch.setattr(cache.subprocess, "run", generate)
    result = cache.prepare_rank(tmp_path, output, 0, world, Path("generator"))
    before = result.stat().st_mtime_ns
    assert cache.prepare_rank(tmp_path, output, 0, world, Path("generator")) == result
    assert len(calls) == 1 and result.stat().st_mtime_ns == before
    result.write_bytes(b"interrupted")
    cache.prepare_rank(tmp_path, output, 0, world, Path("generator"))
    assert len(calls) == 2 and cache.rank_cache_valid(result, expected)
    assert ("--parity" in calls[0]) == (world == 4)
    assert not cache.rank_cache_valid(result, [["different", "uint8", [4], 0, 4]])


def test_model_view_repairs_link_without_editing_checkpoint(modules, tmp_path):
    _, cache, _ = modules
    source, view = tmp_path / "source", tmp_path / "view"
    source.mkdir()
    (source / "weights").write_bytes(b"immutable")
    (source / "chat_template.jinja").write_bytes(b"original")
    template = tmp_path / "template"
    template.write_bytes(b"creator template")
    cache.model_view(source, view, template)
    (view / "weights").unlink()
    cache.model_view(source, view, template)
    assert (view / "weights").is_symlink()
    assert (view / "weights").read_bytes() == b"immutable"
    assert (view / "chat_template.jinja").read_bytes() == b"creator template"
    assert (source / "chat_template.jinja").read_bytes() == b"original"


def test_rank_launch_binds_authorized_fabric_and_shared_pool(modules):
    world, _, launcher = modules
    env = {
        "VONK_TF_WORLD": str(world),
        "VONK_MODEL_REVISION": "a" * 40,
        "VONK_MASTER_ADDR": "192.0.2.1",
        "VONK_MASTER_PORT": "29500",
        "NCCL_IB_HCA": "=rocep1s0:1,rocep2s0:1",
        "NCCL_IB_GID_INDEX": "3",
    }
    args = [
        "--tp",
        str(world),
        "--rank",
        "1",
        "--master",
        "VONK_MASTER_ADDR",
        "--master-port",
        "VONK_MASTER_PORT",
        "--context",
        "1048576",
        "--parallel",
        "4",
    ]
    _, _, command, runtime = launcher.launch(args, env)
    assert command.count("--pool") == 1
    assert command[command.index("--pool") + 1] == str(
        1048576 * (2 if world == 4 else 1)
    )
    assert runtime["NCCL_IB_HCA"] == env["NCCL_IB_HCA"]
    assert runtime["TF_RDMA_DEVICES"] == "rocep1s0,rocep2s0"
    assert runtime["TF_RDMA_GID_INDEX"] == runtime["TF_RDMA_GID"] == "3"
    assert command[5:7] == ["192.0.2.1", "29500"]
    for bad in ("", "mlx5_0:2", "mlx5_0:1,mlx5_0:1", "mlx5_0:1;false"):
        with pytest.raises(ValueError, match="RoCE rails"):
            launcher.launch(args, {**env, "NCCL_IB_HCA": bad})
