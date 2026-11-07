"""Mia AI Lab's 4x DGX Spark TensorFold lane: the launch start.sh and scripts/config.sh run, bound to the Controller."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from vonk_forge_contracts.recipe import RecipeGenerationSettings

from qualification import documents

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/deepseek/mia-1017-tensorfold-four"
RECIPE = "deepseek-v4-1-flash-1017-mia-tensorfold-four"
ENVIRON = {
    "VONK_TF_WORLD": "4",
    "VONK_MODEL_REVISION": "a" * 40,
    "VONK_MASTER_ADDR": "192.0.2.1",
    "VONK_MASTER_PORT": "29500",
    "NCCL_IB_HCA": "=rocep1s0f0:1,roceP2p1s0f0:1",
    "NCCL_IB_GID_INDEX": "3",
}


@pytest.fixture
def modules(monkeypatch):
    loaded = []
    for name in ("prepare_cache", "launcher"):
        spec = importlib.util.spec_from_file_location(name, ADAPTER / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def recipe_arguments(rank):
    recipe = documents.recipe(RECIPE)
    generation = recipe.settings
    assert isinstance(generation, RecipeGenerationSettings)
    settings = {
        "concurrency": generation.concurrency,
        "context_tokens": generation.context_tokens,
    }
    arguments = ["--tp", "4", "--rank", str(rank), "--master", "VONK_MASTER_ADDR"]
    arguments += ["--master-port", "VONK_MASTER_PORT"]
    for item in recipe.runtime.arguments:
        setting = settings[item.setting] if item.setting else None
        value = item.value if setting is None else setting.value
        if value is not False:
            arguments.append("--" + item.name)
            if value is not True:
                arguments.append(str(value))
    return recipe, arguments


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_every_rank_runs_the_kit_settings_on_the_controller_fabric(modules, rank):
    _, launcher = modules
    recipe, arguments = recipe_arguments(rank)
    _, cache, command, env = launcher.launch(arguments, ENVIRON)
    pool = next(a.value for a in recipe.runtime.arguments if a.name == "pool")
    # the README's served pool (PARALLEL 16 x CONTEXT 1,048,576), not scripts/config.sh's 2,097,152
    assert pool == 16 * 1_048_576
    assert env["TF_DS_WINDOW"] == str(pool) and env["TF_DS_STREAMS"] == "16"
    assert env["TF_RDMA_DEVICES"] == "rocep1s0f0,roceP2p1s0f0"
    assert env["TF_DS_RDMA_GID"] == "3" and env["NCCL_IB_HCA"] == ENVIRON["NCCL_IB_HCA"]
    assert (
        env["TF_DS_ROUND_ROWS"] == "48" and len(env["TF_DS_ROUND_MS"].split(",")) == 48
    )
    assert env["TF_DS_DRAFT_MS"] == "25" and env["TF_DS_SERVED_K"] == "solo"
    assert env["TF_DS_2D_GU"] == "parity" and env["TF_DS_2D_ORDER"] == "0,1,2,3"
    assert env["NCCL_CROSS_NIC"] == "1" and env["NCCL_MAX_NCHANNELS"] == "4"
    assert env["NCCL_P2P_DISABLE"] == env["NCCL_SHM_DISABLE"] == "1"
    # the Controller's own NCCL settings are never replaced by the kit's
    assert (
        launcher.launch(arguments, {**ENVIRON, "NCCL_CROSS_NIC": "0"})[3][
            "NCCL_CROSS_NIC"
        ]
        == "0"
    )
    if rank == 0:
        assert command[1:3] == ["serve", str(cache / "model")]
        assert command[command.index("--backend") + 1] == "cuda"
        assert command[command.index("--context") + 1] == "1048576"
        assert command[command.index("--parallel") + 1] == "16"
        assert command[command.index("--reasoning-effort") + 1] == "max"
        assert "--pool" not in command and "--thinking" in command
        assert "--no-drafts" not in command
        assert env["TF_TP_MASTER"] == "192.0.2.1" and env["TF_TP_PORT"] == "29500"
        assert env["TF_DS_ENGRAM"] == "/models/target" and "TF_TP_LISTEN" not in env
        assert env["TENSORFOLD_CUDA_KERNELS"] == str(cache / "kit/aot")
    else:
        assert command[1:8] == [
            "/models/target",
            str(cache / "rank-cache"),
            str(rank),
            "4",
            "192.0.2.1",
            "29500",
            str(cache / "kit"),
        ]
        assert command[command.index("--pool") + 1] == str(pool)
        assert command[command.index("--context") + 1] == "1048576"
        assert command[command.index("--rdma") + 1] == env["TF_RDMA_DEVICES"]
        assert command[command.index("--engram") + 1] == "/models/target"
        assert command[command.index("--drafts") + 1] == "1"
        assert command[command.index("--streams") + 1] == "16"
        assert command[command.index("--round-rows") + 1] == "48"
        assert "TF_TP_MASTER" not in env


def test_unauthorized_fabric_and_a_pool_below_one_context_refuse(modules):
    _, launcher = modules
    _, arguments = recipe_arguments(1)
    for bad in ("", "mlx5_0:2", "mlx5_0:1,mlx5_0:1", "mlx5_0:1;false"):
        with pytest.raises(ValueError, match="RoCE rails"):
            launcher.launch(arguments, {**ENVIRON, "NCCL_IB_HCA": bad})
    with pytest.raises(ValueError, match="GID"):
        launcher.launch(arguments, {**ENVIRON, "NCCL_IB_GID_INDEX": "x"})
    small = [*arguments[:-1], "1000"]
    with pytest.raises(ValueError, match="full-context"):
        launcher.launch(small, ENVIRON)


def test_the_node_weight_file_is_baked_in_the_parity_layout(
    modules, tmp_path, monkeypatch
):
    cache, _ = modules
    calls = []

    def bake(command, check=False):
        calls.append(command)
        raise RuntimeError("stop")

    monkeypatch.setattr(cache.subprocess, "check_output", lambda command: b"[]")
    monkeypatch.setattr(cache.subprocess, "run", bake)
    with pytest.raises(RuntimeError):
        cache.prepare_rank(tmp_path, tmp_path / "rank-cache", 2, 4, Path("generator"))
    assert calls[0][-1] == "--parity" and calls[0][3:5] == ["2", "4"]


def test_the_kit_tools_make_the_rope_tables_and_the_bias_from_the_checkpoint(
    modules, tmp_path, monkeypatch
):
    cache, _ = modules
    source = tmp_path / "kit-source"
    (source / "vision").mkdir(parents=True)
    (source / "VONK_SOURCE_SHA256").write_text("kit")
    (source / "rope.json").write_text(
        json.dumps({"rope.f32": hashlib.sha256(b"rope").hexdigest()})
    )
    monkeypatch.setattr(cache, "BIAS_SHA256", hashlib.sha256(b"bias").hexdigest())
    made = []
    payload = {"make_rope.py": b"rope", "make_bias_vl.py": b"bias"}

    def make(command, check):
        tool = Path(command[1]).name
        made.append((tool, command[2]))
        target = (
            Path(command[3]) / "rope.f32"
            if tool == "make_rope.py"
            else Path(command[3])
        )
        target.write_bytes(payload[tool])

    monkeypatch.setattr(cache.subprocess, "run", make)
    destination = tmp_path / "kit"
    cache.prepare_kit(tmp_path / "checkpoint", destination, source, tmp_path)
    assert made == [
        ("make_rope.py", str(tmp_path / "checkpoint")),
        ("make_bias_vl.py", str(tmp_path / "checkpoint")),
    ]
    records = json.loads((destination / "vonk-generated.json").read_text())
    assert (
        records["rope.f32"]["size"] == 4
        and records["vision/gate_bias_vl.safetensors"]["size"] == 4
    )
    cache.prepare_kit(tmp_path / "checkpoint", destination, source, tmp_path)
    assert len(made) == 2
    payload["make_bias_vl.py"] = b"other"
    (destination / "vonk-generated.json").unlink()
    with pytest.raises(ValueError, match="kit digest differs"):
        cache.prepare_kit(tmp_path / "checkpoint", destination, source, tmp_path)


def test_image_builds_the_engine_and_kit_the_lock_names():
    lock = json.loads((ADAPTER / "kit-lock.json").read_text())
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    engine = lock["dependencies"]["engine"]["commit"]
    kit = lock["dependencies"]["kernel-kit"]["commit"]
    assert f"TensorFold/tar.gz/{engine}" in dockerfile
    assert f'ENGINE_COMMIT)" = {engine}' in dockerfile
    assert f"deepseek-v4.1-tensorfold-tp4-4xgb10/tar.gz/{kit}" in dockerfile
    assert lock["kit"]["commit"] in dockerfile
    for patch in sorted((ADAPTER / "zig/patches").glob("*.patch")):
        assert patch.stat().st_size


def test_kernel_scan_is_the_shared_scan():
    shared = ROOT / "adapters/nemotron/mia-bolt-tensorfold-single/scan-sm121.py"
    assert (ADAPTER / "scan-sm121.py").read_bytes() == shared.read_bytes()
