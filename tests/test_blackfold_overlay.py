"""BLACKFOLD serves the creator's overlay tensors, never the stock pack's, and never a stock rank cache."""

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.documents import model, recipe

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/deepseek/blackfold-1017-tensorfold-tp2"
SLUG = "deepseek-v4-1-flash-1017-blackfold-tensorfold-dual"


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


def test_model_view_links_the_overlay_after_every_stock_shard(modules, tmp_path):
    cache, _ = modules
    target, overlay = tmp_path / "target", tmp_path / "overlay"
    target.mkdir()
    overlay.mkdir()
    for name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        (target / name).write_bytes(b"stock")
    (overlay / cache.OVERLAY_FILE).write_bytes(b"overlay")
    template = tmp_path / "template"
    template.write_bytes(b"template")
    view = tmp_path / "view"
    cache.model_view(target, view, template, overlay / cache.OVERLAY_FILE)
    # The loader maps *.safetensors in byte order and the last file naming a tensor wins.
    shards = sorted(p.name for p in view.iterdir() if p.name.endswith(".safetensors"))
    assert shards[-1] == cache.OVERLAY_LINK
    assert (view / cache.OVERLAY_LINK).read_bytes() == b"overlay"


def test_no_stock_shard_sorts_after_the_overlay_link(modules):
    cache, _ = modules
    primary = next(m for m in recipe(SLUG).models if m.id == "primary")
    stock = model(primary.model.slug)
    paths = [f.path for f in stock.files if f.path.endswith(".safetensors")]
    assert paths and all(path < cache.OVERLAY_LINK for path in paths)


def test_rank_caches_are_keyed_by_the_overlay_revision(modules):
    _, launcher = modules
    definition = recipe(SLUG)
    overlay = next(m for m in definition.models if m.id == "overlay")
    revision = model(overlay.model.slug).model_dump()["source"]["revision"]
    environment = {e.name: e.value for e in definition.runtime.environment}
    assert environment["VONK_MODEL_REVISION"] == revision
    arguments = [
        "--tp", "2", "--rank", "1", "--master", "VONK_MASTER_ADDR",
        "--master-port", "VONK_MASTER_PORT",
        "--context", str(definition.model_dump()["settings"]["context_tokens"]["value"]), "--parallel", "4",
    ]  # fmt: skip
    env = {
        "VONK_TF_WORLD": "2",
        "VONK_MODEL_REVISION": revision,
        "VONK_MASTER_ADDR": "192.0.2.1",
        "VONK_MASTER_PORT": "29500",
        "NCCL_IB_HCA": "=rocep1s0:1,rocep2s0:1",
        "NCCL_IB_GID_INDEX": "3",
    }
    _, cache, command, _ = launcher.launch(arguments, env)
    assert revision in cache.parts
    assert command[command.index("--pool") + 1] == "1048576"
