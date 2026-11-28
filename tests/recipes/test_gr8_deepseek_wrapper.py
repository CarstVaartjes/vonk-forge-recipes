"""The DeepSeek gr8 placement wrapper maps the platform's environment onto the creator's knobs and gr8's entrypoint."""

import os
import subprocess
from pathlib import Path

import pytest
from repo_root import ROOT

ADAPTER = ROOT / "adapters/deepseek/kindlingai-gr8-ds41f-four"

FAKE_KNOBS = """KNOBS_SAW_EXTRA_ENV=$EXTRA_ENV
export KNOBS_SAW_EXTRA_ENV
"""
FAKE_ENTRYPOINT = """#!/bin/bash
env > "$DUMP"
echo "$@" > "$DUMP.args"
"""
PLACEMENT = {
    "VONK_RANK": "2",
    "VONK_WORLD_SIZE": "4",
    "VONK_LOCAL_ADDR": "10.9.0.3",
    "VONK_MASTER_ADDR": "10.9.0.1",
    "VONK_MASTER_PORT": "29555",
    "VONK_PEER_ADDRS": "10.9.0.1,10.9.0.2,10.9.0.3,10.9.0.4",
}


@pytest.fixture
def sandbox(tmp_path):
    """The wrapper with the container's absolute paths moved under tmp_path."""
    creator = tmp_path / "opt/ds41-4x"
    creator.mkdir(parents=True)
    (creator / "knobs.sh").write_text(FAKE_KNOBS)
    entrypoint = tmp_path / "opt/gr8/docker/entrypoint.sh"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(FAKE_ENTRYPOINT)
    entrypoint.chmod(0o755)
    script = (ADAPTER / "gr8-serve").read_text()
    for old, new in (
        ("/opt/ds41-4x", f"{tmp_path}/opt/ds41-4x"),
        ("/opt/gr8", f"{tmp_path}/opt/gr8"),
        ("/outputs", f"{tmp_path}/outputs"),
        ("/models", f"{tmp_path}/models"),
    ):
        script = script.replace(old, new)
    wrapper = tmp_path / "gr8-serve"
    wrapper.write_text(script)
    wrapper.chmod(0o755)

    def run(env, *args):
        full = {"PATH": os.environ["PATH"], "DUMP": str(tmp_path / "dump"), **env}
        done = subprocess.run(
            [str(wrapper), *args], env=full, capture_output=True, text=True, check=False
        )
        dump = {}
        if (tmp_path / "dump").exists():
            for line in (tmp_path / "dump").read_text().splitlines():
                key, _, value = line.partition("=")
                dump[key] = value
        return done, dump

    return tmp_path, run


def test_placement_paths_and_settings(sandbox):
    root, run = sandbox
    done, dump = run(
        {**PLACEMENT, "IPS": "x", "IPS2": "y", "SHAPE": "ring"},
        "--host",
        "0.0.0.0",
        "--port",
        "8100",
        "--cap",
        "327680",
        "--sess",
        "32",
    )
    assert done.returncode == 0, done.stderr
    home = f"{root}/outputs/cache/gr8/home"
    assert (root / "dump.args").read_text().split() == ["up"]
    assert dump["MODEL"] == f"{root}/models/target"
    assert (dump["PORT"], dump["CAP"], dump["SESS"]) == ("8100", "327680", "32")
    assert dump["PEERS"] == PLACEMENT["VONK_PEER_ADDRS"]
    assert (dump["NODE_IP"], dump["CONTROL_PORT"], dump["MENTAT"]) == (
        "10.9.0.3",
        "29555",
        "0",
    )
    assert not {"IPS", "IPS2", "SHAPE"} & dump.keys()
    assert dump["GR8_WORK_DIR"] == home
    assert dump["ENGRAM_PREFIX"] == f"{home}/engram/ds41"
    assert Path(f"{home}/engram").is_dir()
    # the creator's knobs see the gen cache and nothing else
    assert dump["KNOBS_SAW_EXTRA_ENV"].split() == [f"GR8_GEN_CACHE={home}/gen-cache"]


def test_missing_placement_and_unknown_arguments_are_refused(sandbox):
    _, run = sandbox
    done, dump = run({})
    assert done.returncode != 0
    assert not dump
    done, _ = run(PLACEMENT, "--surprise", "1")
    assert done.returncode == 2
