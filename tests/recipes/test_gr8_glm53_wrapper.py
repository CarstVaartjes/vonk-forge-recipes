"""The GLM-5.3 gr8 placement wrapper maps the platform's environment onto gr8's image entrypoint."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from repo_root import ROOT

GLM = ROOT / "adapters/glm"
ADAPTERS = tuple(sorted(path.name for path in GLM.glob("kindlingai-gr8-glm53-*")))
# the wrappers that convert a mounted DFlash2 drafter; the MTP-only ones (five, six) carry no drafter block
DRAFTER_ADAPTERS = tuple(
    name for name in ADAPTERS if "DFLASH_DIR=" in (GLM / name / "gr8-serve").read_text()
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


def placement(world, rank):
    peers = ",".join(f"10.9.0.{i + 1}" for i in range(world))
    return {
        "VONK_RANK": str(rank),
        "VONK_WORLD_SIZE": str(world),
        "VONK_LOCAL_ADDR": f"10.9.0.{rank + 1}",
        "VONK_MASTER_ADDR": "10.9.0.1",
        "VONK_MASTER_PORT": "29555",
        "VONK_PEER_ADDRS": peers,
    }


def make_sandbox(name, tmp_path):
    """The wrapper with the container's absolute paths moved under tmp_path."""
    script = (GLM / name / "gr8-serve").read_text()
    gr8 = tmp_path / "opt/gr8"
    (gr8 / "docker").mkdir(parents=True)
    (gr8 / "tools/launch").mkdir(parents=True)
    (gr8 / "docker/entrypoint.sh").write_text(FAKE_ENTRYPOINT)
    (gr8 / "gr8").write_text(FAKE_GR8)
    (gr8 / "tools/launch/hfsource.py").write_text(FAKE_HFSOURCE)
    for path in (gr8 / "docker/entrypoint.sh", gr8 / "gr8"):
        path.chmod(0o755)
    (tmp_path / "models/drafter").mkdir(parents=True)
    for old, new in (
        ("/opt/gr8", f"{tmp_path}/opt/gr8"),
        ("/outputs", f"{tmp_path}/outputs"),
        ("/models", f"{tmp_path}/models"),
    ):
        script = script.replace(old, new)
    wrapper = tmp_path / "gr8-serve"
    wrapper.write_text(script)
    wrapper.chmod(0o755)

    def run(env, *args):
        (tmp_path / "dump").unlink(missing_ok=True)
        full = {
            "PATH": os.environ["PATH"],
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

    return tmp_path, run


@pytest.fixture(params=ADAPTERS)
def sandbox(request, tmp_path):
    return make_sandbox(request.param, tmp_path)


@pytest.fixture(params=DRAFTER_ADAPTERS)
def drafter_sandbox(request, tmp_path):
    return make_sandbox(request.param, tmp_path)


def test_both_wrapper_shapes_are_covered():
    assert DRAFTER_ADAPTERS
    assert set(ADAPTERS) - set(DRAFTER_ADAPTERS)


def test_placement_paths_and_drafter(drafter_sandbox):
    root, run = drafter_sandbox
    state = f"{root}/outputs/cache/gr8"
    env = {
        **placement(4, 2),
        "MODEL": "https://huggingface.co/x/y@" + "0" * 40,
        "DRAFTER": "https://huggingface.co/a/b@" + "1" * 40,
        "DFLASH": "1",
        "QO_CACHE": f"{state}/cache/qocache",
    }
    done, dump = run(
        env,
        "--host",
        "0.0.0.0",
        "--port",
        "8100",
        "--max-model-len",
        "1048576",
        "--sess",
        "8",
    )
    assert done.returncode == 0, done.stderr
    assert dump["MODEL"] == f"{root}/models/target"
    assert "DRAFTER" not in dump
    assert (dump["PORT"], dump["MAX_MODEL_LEN"], dump["SESS"]) == (
        "8100",
        "1048576",
        "8",
    )
    assert (dump["PEERS"], dump["NODE_IP"], dump["CONTROL_PORT"]) == (
        "10.9.0.1,10.9.0.2,10.9.0.3,10.9.0.4",
        "10.9.0.3",
        "29555",
    )
    assert "IPS" not in dump and "IPS2" not in dump
    assert dump["GR8_WORK_DIR"] == f"{state}/home"
    assert dump["MODEL_CACHE"] == f"{state}/cache"
    assert Path(state, "cache/qocache").is_dir()
    assert dump["DFLASH_DIR"] == f"{state}/cache/dflash/tp4"
    complete = json.loads(Path(dump["DFLASH_DIR"], "rank2.COMPLETE").read_text())
    assert (complete["src"], complete["tp"]) == (f"{root}/models/drafter", 4)


def test_mtp_only_converts_nothing(sandbox):
    root, run = sandbox
    done, dump = run({**placement(6, 1), "TP": "6", "DFLASH": "0"})
    assert done.returncode == 0, done.stderr
    assert "DFLASH_DIR" not in dump
    assert not Path(root, "outputs/cache/gr8/cache/dflash").exists()


def test_tp_from_the_recipe_names_the_drafter_pack(drafter_sandbox):
    _, run = drafter_sandbox
    _, dump = run({**placement(6, 1), "TP": "6"})
    assert dump["DFLASH_DIR"].endswith("/cache/dflash/tp6")


def test_compact_copies_are_prepared_once(sandbox):
    root, run = sandbox
    cache = root / "outputs/cache/gr8/cache"
    stale = cache / ".compact-tp4.partial-1234"
    stale.mkdir(parents=True)
    (stale / "half").write_text("x")
    env = {
        **placement(4, 1),
        "DFLASH": "0",
        "EXTRA_ENV": f"GR8_QUIT_FAST=1 GR8_LOAD_COMPACT={cache}/compact-tp4 GR8_LOAD_MERGE_KB=1",
        "COMPACT_EXPERTS": "1",
    }
    done, dump = run(env)
    assert done.returncode == 0, done.stderr
    assert not stale.exists()
    assert "COMPACT_EXPERTS" not in dump
    assert (root / "prepare.log").read_text().splitlines() == ["glm53 prepare --rank 1"]
    run(env)
    run({**env, "COMPACT_EXPERTS": "0"})
    assert len((root / "prepare.log").read_text().splitlines()) == 1


def test_compact_copies_need_a_directory(sandbox):
    root, run = sandbox
    done, dump = run({**placement(4, 0), "DFLASH": "0", "COMPACT_EXPERTS": "1"})
    assert done.returncode == 2
    assert not dump and not (root / "prepare.log").exists()


def test_unknown_argument_is_refused(sandbox):
    _, run = sandbox
    done, dump = run({}, "--surprise", "1")
    assert done.returncode == 2
    assert not dump
