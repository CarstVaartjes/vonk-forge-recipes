"""The gr8 Sage placement wrapper maps the platform's environment onto gr8's entrypoint and the creator's settings."""

import os
import subprocess
from pathlib import Path

import pytest
from repo_root import ROOT

ADAPTERS = {"kindlingai-gr8-sage-dual": 2, "kindlingai-gr8-sage-four": 4}

FAKE_ENTRYPOINT = """#!/bin/bash
env > "$DUMP"
"""
FAKE_GR8 = """#!/bin/bash
echo "$@" >> "$PREPARE_LOG"
"""


@pytest.fixture(params=sorted(ADAPTERS))
def sandbox(request, tmp_path):
    """The wrapper with the container's absolute paths moved under tmp_path."""
    name = request.param
    tp = ADAPTERS[name]
    script = (ROOT / "adapters/sage" / name / "gr8-serve").read_text()
    gr8 = tmp_path / "opt/gr8"
    (gr8 / "docker").mkdir(parents=True)
    (gr8 / "docker/entrypoint.sh").write_text(FAKE_ENTRYPOINT)
    (gr8 / "gr8").write_text(FAKE_GR8)
    for path in (gr8 / "docker/entrypoint.sh", gr8 / "gr8"):
        path.chmod(0o755)
    (tmp_path / "etc/gr8").mkdir(parents=True)
    (tmp_path / "etc/gr8/kv-store.yaml").write_text(
        "ssd: {path: /var/lib/gr8/cache/prompts}\n"
    )
    for old in ("/opt/gr8", "/outputs", "/etc/gr8", "/models"):
        script = script.replace(old, f"{tmp_path}{old}")
    wrapper = tmp_path / "gr8-serve"
    wrapper.write_text(script)
    wrapper.chmod(0o755)
    placement = {
        "VONK_RANK": str(tp - 1),
        "VONK_WORLD_SIZE": str(tp),
        "VONK_LOCAL_ADDR": f"10.9.0.{tp}",
        "VONK_MASTER_ADDR": "10.9.0.1",
        "VONK_MASTER_PORT": "29555",
        "VONK_PEER_ADDRS": ",".join(f"10.9.0.{i}" for i in range(1, tp + 1)),
    }

    def run(env, *args):
        full = {
            "PATH": os.environ["PATH"],
            "TP": str(tp),
            "DUMP": str(tmp_path / "dump"),
            "PREPARE_LOG": str(tmp_path / "prepare.log"),
            **placement,
            **env,
        }
        done = subprocess.run(
            [str(wrapper), *args], env=full, capture_output=True, text=True, check=False
        )
        dump = {}
        if (tmp_path / "dump").exists():
            for line in (tmp_path / "dump").read_text().splitlines():
                key, _, value = line.partition("=")
                dump[key] = value
        return done, dump

    return tmp_path, tp, placement, run


def test_placement_paths_and_model_in_place(sandbox):
    root, tp, placement, run = sandbox
    env = {"MODEL": "https://huggingface.co/x/y@" + "0" * 40, "IPS": "10.0.0.1"}
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
    assert (dump["PORT"], dump["MAX_MODEL_LEN"], dump["SESS"]) == (
        "8100",
        "262144",
        "4",
    )
    assert dump["GR8_WORK_DIR"] == f"{state}/home"
    assert dump["MODEL_CACHE"] == f"{state}/cache"
    assert dump["KV_STORE"] == f"@{state}/kv-store.yaml"
    assert f"path: {state}/prompts" in Path(f"{state}/kv-store.yaml").read_text()
    assert dump["PEERS"] == placement["VONK_PEER_ADDRS"]
    assert len(dump["PEERS"].split(",")) == tp
    assert dump["NODE_IP"] == placement["VONK_LOCAL_ADDR"]
    assert dump["CONTROL_PORT"] == "29555"
    assert "IPS" not in dump
    assert "IPS2" not in dump
    # text only unless the vision option asks: gr8's own default, nothing added
    assert "EXTRA_ENV" not in dump


def test_vision_and_compact_copies_compose_one_extra_env(sandbox):
    root, tp, _, run = sandbox
    env = {"VISION": "1", "COMPACT_EXPERTS": "1"}
    done, dump = run(env)
    assert done.returncode == 0, done.stderr
    assert dump["EXTRA_ENV"].split() == [
        "GR8_GLM53F_VISION=1",
        f"GR8_LOAD_COMPACT={root}/outputs/cache/gr8/cache/compact-tp{tp}",
        "GR8_LOAD_MERGE_KB=1",
    ]
    assert "VISION" not in dump
    assert "COMPACT_EXPERTS" not in dump
    assert (root / "prepare.log").read_text().splitlines() == [
        f"glm53f prepare --rank {tp - 1}"
    ]
    # a finished prepare is reused: the next start writes no copies
    run(env)
    assert len((root / "prepare.log").read_text().splitlines()) == 1


def test_context_parallel_joins_the_extra_env(sandbox):
    _, _, _, run = sandbox
    done, dump = run({"CONTEXT_PARALLEL": "1", "VISION": "1"})
    assert done.returncode == 0, done.stderr
    assert dump["EXTRA_ENV"].split() == ["GR8_GLM53F_VISION=1", "GR8_CP=1"]
    assert "CONTEXT_PARALLEL" not in dump


def test_an_interrupted_prepare_is_redone(sandbox):
    root, tp, _, run = sandbox
    stale = root / f"outputs/cache/gr8/cache/.compact-tp{tp}.partial-1234"
    stale.mkdir(parents=True)
    (stale / "half").write_text("x")
    done, _ = run({"COMPACT_EXPERTS": "1"})
    assert done.returncode == 0, done.stderr
    assert not stale.exists()
    assert (root / "prepare.log").read_text().startswith("glm53f prepare --rank")


def test_unknown_argument_is_refused(sandbox):
    _, _, _, run = sandbox
    done, dump = run({}, "--surprise", "1")
    assert done.returncode == 2
    assert not dump
