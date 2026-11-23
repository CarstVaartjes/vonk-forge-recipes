"""The derived target view must keep a working link to the overlay shard after the staging directory is renamed."""

import os
import textwrap
from pathlib import Path
from typing import Any

from repo_root import ROOT

ADAPTER = ROOT / "adapters/glm/tonyd2wild-knapcio-tp2-dual"

# Stands in for scripts/glm_quant_mix.py: plan and quant write the overlay, assemble links like the real one (os.link).
FAKE_QUANT_MIX = textwrap.dedent(
    """
    import json, os, sys
    command, arguments = sys.argv[1], dict(zip(sys.argv[2::2], sys.argv[3::2]))
    if command == "plan":
        open(arguments["--out"], "w").write("{}")
    elif command == "quant":
        os.makedirs(arguments["--out"])
        open(os.path.join(arguments["--out"], "model-qmix-lossless8.safetensors"), "wb").write(b"overlay")
        json.dump({"overlay": "model-qmix-lossless8.safetensors"}, open(os.path.join(arguments["--out"], "manifest.json"), "w"))
    else:
        os.makedirs(arguments["--dst"])
        os.link(os.path.join(arguments["--base"], "shard.safetensors"), os.path.join(arguments["--dst"], "shard.safetensors"))
        overlay = json.load(open(os.path.join(arguments["--overlay"], "manifest.json")))
        os.link(os.path.join(arguments["--overlay"], overlay["overlay"]), os.path.join(arguments["--dst"], overlay["overlay"]))
        for name in ("config.json", "model.safetensors.index.json"):
            open(os.path.join(arguments["--dst"], name), "w").write("{}")
        json.dump(overlay, open(os.path.join(arguments["--dst"], "qmix-manifest.json"), "w"))
    """
)
FAKE_DRAFTER = (
    "import os, sys\nos.makedirs(sys.argv[2])\n"
    "for name in ('model.safetensors', 'config.json'):\n    open(os.path.join(sys.argv[2], name), 'wb').write(b'draft')\n"
)


def _wrapper(tmp_path):
    source = (ADAPTER / "vllm-wrapper.py").read_text()
    namespace: dict[str, Any] = {"__name__": "vllm_wrapper"}
    # Only the definitions: the rest of the file validates the launch and replaces the process.
    head = source.split("\narguments = sys.argv[1:]")[0]
    exec(compile(head, "vllm-wrapper.py", "exec"), namespace)  # noqa: S102
    models, scripts = tmp_path / "models", tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "glm_quant_mix.py").write_text(FAKE_QUANT_MIX)
    (scripts / "drafter_fp8.py").write_text(FAKE_DRAFTER)
    for directory, files in (
        (
            "target",
            ("config.json", "model.safetensors.index.json", "hf_quant_config.json"),
        ),
        ("drafter", ("config.json",)),
    ):
        (models / directory).mkdir(parents=True)
        for name in files:
            (models / directory / name).write_text("{}")
    (models / "target/shard.safetensors").write_bytes(b"shard")
    (models / "drafter/model.safetensors").write_bytes(b"draft")
    namespace.update(
        MODELS=models,
        TARGET=models / "target",
        DRAFTER=models / "drafter",
        SCRIPTS=scripts,
        CACHE=tmp_path / "cache",
    )
    namespace["VIEW_KEY_FILES"] = (
        (models / "target", "config.json"),
        (models / "target", "model.safetensors.index.json"),
        (models / "target", "hf_quant_config.json"),
        (models / "drafter", "config.json"),
    )
    return namespace, models


def test_view_keeps_overlay_link_after_rename_and_is_reused(tmp_path, monkeypatch):
    wrapper, _ = _wrapper(tmp_path)
    runs = []
    real_run = wrapper["_run"]
    monkeypatch.setitem(
        wrapper, "_run", lambda *command: (runs.append(command), real_run(*command))
    )
    wrapper["_prepare_views"]()
    lossless8 = Path(wrapper["CACHE"]) / "target"
    # Model shards sit on a read-only mount: linked by symlink. The overlay shard sits beside the view: a real hardlink.
    assert (lossless8 / "shard.safetensors").is_symlink()
    assert (lossless8 / "shard.safetensors").read_bytes() == b"shard"
    overlay = lossless8 / "model-qmix-lossless8.safetensors"
    assert not overlay.is_symlink()
    assert overlay.read_bytes() == b"overlay"
    assert os.stat(overlay).st_nlink == 2
    built = len(runs)
    wrapper["_prepare_views"]()
    assert len(runs) == built
    # A view left with a dangling overlay link by an older build is rebuilt, not reused.
    view = lossless8.resolve().parent
    overlay.unlink()
    overlay.symlink_to(view / "gone")
    assert not wrapper["_complete"](view)
    wrapper["_prepare_views"]()
    assert len(runs) > built
    assert wrapper["_complete"](view)
