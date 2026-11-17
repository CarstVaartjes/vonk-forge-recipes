"""BLACKFOLD serves the creator's overlay tensors, never the stock pack's, and never a stock rank cache."""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

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


def checkpoint(root, overlay_names):
    """A stock folder with two shards and an index, and the overlay folder."""
    target, overlay = root / "target", root / "overlay"
    target.mkdir()
    overlay.mkdir()
    for name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        (target / name).write_bytes(b"stock")
    weight_map = {"a.wo_b.trellis": "model-00001-of-00002.safetensors"}
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    header = json.dumps({"__metadata__": {}, **{n: {} for n in overlay_names}}).encode()
    path = overlay / "heretic_wo_b_exl3_k5_patched.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    template = root / "template"
    template.write_bytes(b"template")
    return target, overlay, template


def test_overlay_loads_after_every_stock_shard_and_the_rank_cache_reads_that_view(
    modules, tmp_path, monkeypatch
):
    cache, _ = modules
    target, overlay, template = checkpoint(tmp_path, ["a.wo_b.trellis"])
    seen = []
    monkeypatch.setattr(cache, "prepare_rank", lambda model, *rest: seen.append(model))
    monkeypatch.setattr(cache.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(cache, "prepare_kit", lambda *a: None)
    out = tmp_path / "cache"
    out.mkdir()
    (out / ".token-map.partial").write_text("[1]")
    cache.prepare(
        target, overlay, out, 0, 2, engine=tmp_path / "engine", template=template
    )
    view = out / "model"
    assert seen == [view]
    # The loader maps *.safetensors in byte order and the last file naming a tensor wins.
    shards = sorted(p.name for p in view.iterdir() if p.name.endswith(".safetensors"))
    assert shards[-1] == cache.OVERLAY_LINK
    assert (view / cache.OVERLAY_LINK).resolve() == overlay / cache.OVERLAY_FILE


def test_an_overlay_that_replaces_nothing_is_refused(modules, tmp_path):
    cache, _ = modules
    target, overlay, template = checkpoint(tmp_path, ["unknown.tensor"])
    with pytest.raises(ValueError, match="do not replace"):
        cache.model_view(
            target, tmp_path / "view", template, overlay / cache.OVERLAY_FILE
        )


def test_rank_caches_are_keyed_by_the_overlay_revision_and_launch_matches_the_recipe(
    modules,
):
    _, launcher = modules
    recipe = json.loads((ROOT / "recipes" / f"{SLUG}.json").read_text())
    overlay = next(m for m in recipe["models"] if m["id"] == "overlay")["model"]
    source = json.loads((ROOT / "models" / f"{overlay['slug']}.json").read_text())[
        "source"
    ]
    environment = {e["name"]: e["value"] for e in recipe["runtime"]["environment"]}
    assert environment["VONK_MODEL_REVISION"] == source["revision"]
    settings = recipe["settings"]
    arguments = [
        "--tp",
        "2",
        "--rank",
        "1",
        "--master",
        "VONK_MASTER_ADDR",
        "--master-port",
        "VONK_MASTER_PORT",
    ]
    for item in recipe["runtime"]["arguments"]:
        value = (
            settings[item["setting"]]["value"] if "setting" in item else item["value"]
        )
        arguments += ["--" + item["name"]] + ([] if value is True else [str(value)])
    env = {
        "VONK_TF_WORLD": "2",
        "VONK_MODEL_REVISION": environment["VONK_MODEL_REVISION"],
        "VONK_MASTER_ADDR": "192.0.2.1",
        "VONK_MASTER_PORT": "29500",
        "NCCL_IB_HCA": "=rocep1s0:1,rocep2s0:1",
        "NCCL_IB_GID_INDEX": "3",
    }
    _, cache, command, _ = launcher.launch(arguments, env)
    assert source["revision"] in cache.parts
    assert command[command.index("--pool") + 1] == str(
        settings["context_tokens"]["value"]
    )
    assert command[command.index("--drafts") + 1] == "1"
