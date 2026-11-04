"""bertholomus's v0.3.6 TP4 lane: the launch serve.sh runs, bound to the Controller, and its readied checkpoint."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from vonk_forge_contracts.recipe import RecipeGenerationSettings

from qualification import documents

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/deepseek/bertholomus-1017-tensorfold-tp4"
RECIPE = "deepseek-v4-1-flash-1017-exl3-4bpw-bertholomus-tensorfold-four"
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
def test_every_rank_runs_the_served_settings_on_the_controller_fabric(modules, rank):
    _, launcher = modules
    recipe, arguments = recipe_arguments(rank)
    _, cache, command, env = launcher.launch(arguments, ENVIRON)
    pool = next(a.value for a in recipe.runtime.arguments if a.name == "pool")
    # tonyd2wild's dedicated-node pool (run README at 74ac0136), the owner's rule
    assert pool == 29_360_128
    assert env["TF_DS_WINDOW"] == str(pool) and env["TF_DS_STREAMS"] == "16"
    assert env["TF_RDMA_DEVICES"] == "rocep1s0f0,roceP2p1s0f0"
    assert env["TF_RDMA_GID"] == "3" and env["NCCL_IB_HCA"] == ENVIRON["NCCL_IB_HCA"]
    assert env["TF_DS_WAVES"] == env["TF_DS_2D_YX"] == env["TF_FILL_MANY"] == "1"
    assert env["NCCL_MAX_NCHANNELS"] == "4" and env["TF_DS_IDX_SPLIT"] == "8192"
    # v0.3.6 serves with its sampling opt-ins unset
    assert "TF_SEED_RANDOM" not in env and "TF_TOP_K_MAX" not in env
    assert env["TENSORFOLD_GRAMMAR_LIB"] == str(cache / "kit/grammar/libtfgrammar.so")
    if rank == 0:
        assert command[1:3] == ["serve", str(cache / "model")]
        assert command[-2:] == ["--backend", "cuda"]
        assert command[command.index("--prompt-cache-gib") + 1] == "2"
        assert command[command.index("--context") + 1] == "1048576"
        assert "--pool" not in command and "--thinking" in command
        assert env["TF_DS_RDMA_KERNELS"].endswith("fatbin/dsv41_rdma.fatbin")
        assert env["TF_TP_MASTER"] == "192.0.2.1" and env["TF_TP_PORT"] == "29500"
    else:
        assert command[3:9] == [
            str(rank),
            "4",
            "192.0.2.1",
            "29500",
            str(cache / "kit"),
            "--pool",
        ]
        # The engine keeps the last --pool: serve.sh passes the context first, the window last.
        pools = [command[i + 1] for i, a in enumerate(command) if a == "--pool"]
        assert pools == ["1048576", str(pool)]
        assert command[command.index("--rdma") + 1] == env["TF_RDMA_DEVICES"]
        assert command[command.index("--streams") + 1] == "16"


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


def test_serving_file_is_hashed_once_and_reused_by_receipt(
    modules, tmp_path, monkeypatch
):
    cache, _ = modules
    view = tmp_path / "view"
    view.mkdir()
    (view / "chat_template.jinja").write_bytes(b"template")
    receipt = tmp_path / "vonk-serving.json"
    good = b"serving tensors"
    monkeypatch.setattr(cache, "SERVING_SHA256", hashlib.sha256(good).hexdigest())
    calls = []
    payload = good

    def tool(command, check):
        # tools/serving_1017.py writes <file>.partial and renames it into place
        assert check and command[2] == str(view)
        assert command[3:] == ["--template", str(view / "chat_template.jinja")]
        calls.append(command)
        partial = view / (cache.SERVING_FILE + ".partial")
        partial.write_bytes(payload)
        partial.replace(view / cache.SERVING_FILE)

    monkeypatch.setattr(cache.subprocess, "run", tool)
    cache.serving_file(view, Path("serving_1017.py"), receipt)
    assert len(calls) == 1
    assert json.loads(receipt.read_text())["size"] == len(good)

    def no_hash(*_):
        raise AssertionError("a start re-hashed the stored serving file")

    with monkeypatch.context() as patched:
        patched.setattr(cache.hashlib, "file_digest", no_hash)
        cache.serving_file(view, Path("serving_1017.py"), receipt)
    assert len(calls) == 1
    # a file whose size no longer matches its receipt is made again
    (view / cache.SERVING_FILE).write_bytes(good + b"!")
    cache.serving_file(view, Path("serving_1017.py"), receipt)
    assert len(calls) == 2 and (view / cache.SERVING_FILE).read_bytes() == good
    receipt.unlink()
    payload = b"other bytes"
    with pytest.raises(ValueError, match="recorded digest"):
        cache.serving_file(view, Path("serving_1017.py"), receipt)
    assert not (view / cache.SERVING_FILE).exists() and not receipt.exists()


def test_kit_bias_copy_is_checked_by_size_not_hashed(modules, tmp_path, monkeypatch):
    cache, _ = modules
    source = tmp_path / "kit-source"
    (source / "vision").mkdir(parents=True)
    (source / "VONK_SOURCE_SHA256").write_text("kit")
    (source / "rope.json").write_text(json.dumps({"rope.f32": "0" * 64}))
    bias = tmp_path / "bias.safetensors"
    bias.write_bytes(b"x" * cache.BIAS_SIZE)

    def make_rope(command, check):
        (Path(command[-1]) / "rope.f32").write_bytes(b"rope")

    def no_hash(*_):
        raise AssertionError("prepare_kit hashed a stored file")

    monkeypatch.setattr(cache.subprocess, "run", make_rope)
    monkeypatch.setattr(cache.hashlib, "file_digest", no_hash)
    destination = tmp_path / "kit"
    cache.prepare_kit(tmp_path, destination, source, tmp_path, bias)
    records = json.loads((destination / "vonk-generated.json").read_text())
    assert records["vision/gate_bias_vl.safetensors"] == {
        "sha256": cache.BIAS_SHA256,
        "size": cache.BIAS_SIZE,
    }
    bias.write_bytes(b"short")
    (destination / "vonk-generated.json").unlink()
    with pytest.raises(ValueError, match="wrong size"):
        cache.prepare_kit(tmp_path, destination, source, tmp_path, bias)


def test_rank_cache_is_made_from_the_readied_view(modules, tmp_path, monkeypatch):
    cache, _ = modules
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "chat_template.jinja").write_text("checkpoint template")
    template = tmp_path / "template"
    template.write_text("recipe template")
    seen = {}
    monkeypatch.setattr(
        cache,
        "serving_file",
        lambda view, tool, receipt: seen.setdefault("serving", view),
    )
    monkeypatch.setattr(
        cache, "prepare_rank", lambda model, *rest: seen.setdefault("rank", model)
    )
    monkeypatch.setattr(
        cache,
        "prepare_kit",
        lambda model, *rest: seen.setdefault("kit", (model, rest[-1])),
    )

    def token_map(command, check):
        seen["tokens"] = command[2]
        Path(command[3]).write_text("[1, 2]")

    monkeypatch.setattr(cache.subprocess, "run", token_map)
    view = cache.prepare(
        source, tmp_path / "bias", tmp_path / "cache", 0, 4, template=template
    )
    assert seen == {
        "serving": view,
        "rank": view,
        "tokens": str(view),
        "kit": (view, tmp_path / "bias"),
    }
    assert (view / "chat_template.jinja").read_text() == "recipe template"
    assert (source / "chat_template.jinja").read_text() == "checkpoint template"


def test_image_builds_the_engine_the_kit_lock_names():
    lock = json.loads((ADAPTER / "kit-lock.json").read_text())
    commit = lock["dependencies"]["engine"]["commit"]
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert f"TensorFold/tar.gz/{commit}" in dockerfile
    assert f'ENGINE_COMMIT)" = {commit}' in dockerfile
    assert lock["kit"]["commit"] in dockerfile


def test_kernel_scan_is_the_shared_scan():
    shared = ROOT / "adapters/nemotron/mia-bolt-tensorfold-single/scan-sm121.py"
    assert (ADAPTER / "scan-sm121.py").read_bytes() == shared.read_bytes()
