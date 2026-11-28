"""The gr8 placement wrapper maps the platform's environment onto the creator's entrypoint."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from repo_root import ROOT

ADAPTERS = tuple(
    name
    for name in ("kindlingai-gr8-flash-single", "kindlingai-gr8-flash-dual")
    if (ROOT / "adapters/glm" / name).is_dir()
)

FAKE_ENTRYPOINT = """#!/bin/bash
env > "$DUMP"
"""
FAKE_GR8 = """#!/bin/bash
echo "$@" >> "$PREPARE_LOG"
"""
FAKE_HFSOURCE = """
import json, os

def converter():
    return ['converter']

def convert_drafter(convert, src, out, tp, rank, ident, log, label, scratch):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'rank%d.COMPLETE' % rank), 'w') as f:
        json.dump({'src': src, 'tp': tp, 'scratch': scratch}, f)
"""


@pytest.fixture(params=ADAPTERS)
def sandbox(request, tmp_path):
    """The wrapper with the container's absolute paths moved under tmp_path."""
    name = request.param
    tp = "2" if name.endswith("dual") else "1"
    script = (ROOT / "adapters/glm" / name / "gr8-serve").read_text()
    creator = tmp_path / f"opt/glm53f-{tp}x"
    (creator / "bin").mkdir(parents=True)
    (creator / "entrypoint.sh").write_text(FAKE_ENTRYPOINT)
    (creator / "bin/gr8").write_text(FAKE_GR8)
    launch = tmp_path / "opt/gr8/tools/launch"
    launch.mkdir(parents=True)
    (launch / "hfsource.py").write_text(FAKE_HFSOURCE)
    (tmp_path / "etc/gr8").mkdir(parents=True)
    (tmp_path / "etc/gr8/kv-store.yaml").write_text("path: /var/lib/gr8/cache/prompts\n")
    (tmp_path / "models/drafter").mkdir(parents=True)
    for path in (creator / "entrypoint.sh", creator / "bin/gr8"):
        path.chmod(0o755)
    for old, new in (
        ("/opt/glm53f-?x", f"{tmp_path}/opt/glm53f-?x"),
        ("/opt/gr8", f"{tmp_path}/opt/gr8"),
        ("/outputs", f"{tmp_path}/outputs"),
        ("/etc/gr8", f"{tmp_path}/etc/gr8"),
        ("/models", f"{tmp_path}/models"),
    ):
        script = script.replace(old, new)
    wrapper = tmp_path / "gr8-serve"
    wrapper.write_text(script)
    wrapper.chmod(0o755)

    def run(env, *args):
        full = {
            "PATH": os.environ["PATH"],
            "TP": tp,
            "DUMP": str(tmp_path / "dump"),
            "PREPARE_LOG": str(tmp_path / "prepare.log"),
            **env,
        }
        done = subprocess.run(
            [str(wrapper), *args],
            env=full,
            capture_output=True,
            text=True,
            check=False,
        )
        dump = {}
        if (tmp_path / "dump").exists():
            for line in (tmp_path / "dump").read_text().splitlines():
                key, _, value = line.partition("=")
                dump[key] = value
        return done, dump

    return tmp_path, tp, run


DUAL_PLACEMENT = {
    "VONK_RANK": "1",
    "VONK_WORLD_SIZE": "2",
    "VONK_LOCAL_ADDR": "10.9.0.2",
    "VONK_MASTER_ADDR": "10.9.0.1",
    "VONK_MASTER_PORT": "29555",
    "VONK_PEER_ADDRS": "10.9.0.1,10.9.0.2",
}


def test_placement_and_paths(sandbox):
    root, tp, run = sandbox
    placement = DUAL_PLACEMENT if tp == "2" else {}
    env = {
        **placement,
        "MODEL": "https://huggingface.co/x/y@" + "0" * 40,
        "DRAFTER": "https://huggingface.co/a/b@" + "1" * 40,
        "DFLASH": "1",
    }
    done, dump = run(
        env,
        "--host",
        "0.0.0.0",
        "--port",
        "8100",
        "--max-model-len",
        "262144",
        "--sess",
        "4",
    )
    assert done.returncode == 0, done.stderr
    state = f"{root}/outputs/cache/gr8"
    assert dump["MODEL"] == f"{root}/models/target"
    assert "DRAFTER" not in dump
    assert (dump["PORT"], dump["MAX_MODEL_LEN"], dump["SESS"]) == (
        "8100",
        "262144",
        "4",
    )
    assert dump["GR8_WORK_DIR"] == f"{state}/home"
    assert dump["MODEL_CACHE"] == f"{state}/cache"
    assert dump["KV_STORE"] == f"@{state}/kv-store.yaml"
    assert f"path: {state}/prompts" in Path(f"{state}/kv-store.yaml").read_text()
    assert dump["DFLASH_DIR"] == f"{state}/cache/dflash/tp{tp}"
    rank = "1" if tp == "2" else "0"
    complete = json.loads(Path(dump["DFLASH_DIR"], f"rank{rank}.COMPLETE").read_text())
    assert complete["src"] == f"{root}/models/drafter"
    if tp == "2":
        assert dump["PEERS"] == "10.9.0.1,10.9.0.2"
        assert dump["NODE_IP"] == "10.9.0.2"
        assert dump["CONTROL_PORT"] == "29555"
        assert "IPS" not in dump
        assert "IPS2" not in dump
        assert (
            dump["VARIANT"]
            == f"{root}/models/target/manifests/exlr8-manifest-bit349.json"
        )
    else:
        assert "PEERS" not in dump
        assert "NODE_IP" not in dump


def test_variant_names_and_tune(sandbox):
    root, _, run = sandbox
    _, dump = run({"TUNE": "quality", "DFLASH": "0"})
    assert dump["VARIANT"] == "k4"
    assert "DFLASH_DIR" not in dump
    assert not Path(root, "outputs/cache/gr8/cache/dflash").exists()
    _, dump = run({"VARIANT": "k2", "DFLASH": "0"})
    assert dump["VARIANT"] == "k2"
    _, dump = run({"VARIANT": "bit201", "DFLASH": "0"})
    assert (
        dump["VARIANT"] == f"{root}/models/target/manifests/exlr8-manifest-bit201.json"
    )


def test_compact_copies_are_prepared_once(sandbox):
    root, tp, run = sandbox
    env = {
        **(DUAL_PLACEMENT if tp == "2" else {}),
        "DFLASH": "0",
        "COMPACT_EXPERTS": "1",
    }
    done, dump = run(env)
    assert done.returncode == 0, done.stderr
    extra = dump["EXTRA_ENV"]
    assert f"GR8_LOAD_COMPACT={root}/outputs/cache/gr8/cache/compact-tp{tp}" in extra
    assert "GR8_LOAD_MERGE_KB=1" in extra
    log = (root / "prepare.log").read_text().splitlines()
    assert len(log) == 1
    assert log[0].startswith("glm53f prepare --rank")
    run(env)
    assert len((root / "prepare.log").read_text().splitlines()) == 1
    _, dump = run({**env, "COMPACT_EXPERTS": "0", "VISION": "0"})
    assert dump["EXTRA_ENV"] == "GR8_GLM53F_VISION=0"
    assert len((root / "prepare.log").read_text().splitlines()) == 1


def test_unknown_argument_is_refused(sandbox):
    _, _, run = sandbox
    done, _ = run({}, "--surprise", "1")
    assert done.returncode == 2
    assert "unknown argument" in done.stderr


def test_a_leftover_staging_directory_does_not_stop_prepare(sandbox):
    root, tp, run = sandbox
    cache = root / "outputs/cache/gr8/cache"
    stale = cache / f".compact-tp{tp}.partial-1234"
    stale.mkdir(parents=True)
    (stale / "half").write_text("x")
    placement = DUAL_PLACEMENT if tp == "2" else {}
    done, _ = run({**placement, "DFLASH": "0", "COMPACT_EXPERTS": "1"})
    assert done.returncode == 0, done.stderr
    assert not stale.exists()
    assert (root / "prepare.log").read_text().startswith("glm53f prepare --rank")
