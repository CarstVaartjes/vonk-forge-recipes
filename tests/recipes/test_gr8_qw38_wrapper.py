"""The Qwen3.8-Flash-Next gr8 placement wrapper maps the platform's environment onto gr8's image entrypoint."""

import os
import subprocess
from pathlib import Path

import pytest
from repo_root import ROOT

QWEN = ROOT / "adapters/qwen"
ADAPTERS = tuple(sorted(path.name for path in QWEN.glob("kindlingai-gr8-qw38-*")))

FAKE_ENTRYPOINT = """#!/bin/bash
env > "$DUMP"
"""


def sandbox(name: str, tmp_path: Path):
    """The wrapper with the container's absolute paths moved under tmp_path."""
    script = (QWEN / name / "gr8-serve").read_text()
    (tmp_path / "opt/gr8/docker").mkdir(parents=True)
    entrypoint = tmp_path / "opt/gr8/docker/entrypoint.sh"
    entrypoint.write_text(FAKE_ENTRYPOINT)
    entrypoint.chmod(0o755)
    for old, new in (
        ("/opt/gr8", f"{tmp_path}/opt/gr8"),
        ("/outputs", f"{tmp_path}/outputs"),
        ("/models", f"{tmp_path}/models"),
    ):
        script = script.replace(old, new)
    wrapper = tmp_path / "gr8-serve"
    wrapper.write_text(script)
    wrapper.chmod(0o755)

    def run(env: dict[str, str], *args: str):
        dump = tmp_path / "dump"
        dump.unlink(missing_ok=True)
        done = subprocess.run(
            [str(wrapper), *args],
            env={"PATH": os.environ["PATH"], "DUMP": str(dump), **env},
            capture_output=True,
            text=True,
            check=False,
        )
        seen = {}
        if dump.exists():
            for line in dump.read_text().splitlines():
                key, _, value = line.partition("=")
                seen[key] = value
        return done, seen

    return run


ARGS = (
    "--host",
    "0.0.0.0",
    "--port",
    "8100",
    "--max-model-len",
    "262144",
    "--sess",
    "6",
)


def test_both_kits_ship_one_wrapper():
    assert len(ADAPTERS) == 2
    assert len({(QWEN / name / "gr8-serve").read_bytes() for name in ADAPTERS}) == 1


@pytest.mark.parametrize("name", ADAPTERS)
def test_arguments_paths_and_engine_flags(name, tmp_path):
    run = sandbox(name, tmp_path)
    done, seen = run(
        {
            "EXTRA_ENV": "GR8_QW38_PLE_CACHE_MIB=1024",
            "VISION": "1",
            "NO_MTP": "1",
            "DFL": "0",
        },
        *ARGS,
    )
    assert done.returncode == 0, done.stderr
    assert seen["MODEL"] == f"{tmp_path}/models/target"
    assert (seen["PORT"], seen["MAX_MODEL_LEN"], seen["SESS"]) == (
        "8100",
        "262144",
        "6",
    )
    assert seen["GR8_WORK_DIR"] == f"{tmp_path}/outputs/cache/gr8/home"
    assert seen["MODEL_CACHE"] == f"{tmp_path}/outputs/cache/gr8/cache"
    assert seen["EXTRA_ENV"] == (
        "GR8_QW38_PLE_CACHE_MIB=1024 GR8_QW38_VISION=1 GR8_QW38_NO_MTP=1 GR8_QW38_DFL=0"
    )
    assert not {"VISION", "NO_MTP", "DFL"} & seen.keys()


@pytest.mark.parametrize("name", ADAPTERS)
def test_default_options_add_no_engine_flag(name, tmp_path):
    run = sandbox(name, tmp_path)
    done, seen = run({"EXTRA_ENV": "GR8_QW38_PLE_CACHE_MIB=4096"}, *ARGS)
    assert done.returncode == 0, done.stderr
    assert seen["EXTRA_ENV"] == "GR8_QW38_PLE_CACHE_MIB=4096"


@pytest.mark.parametrize("name", ADAPTERS)
def test_placement(name, tmp_path):
    run = sandbox(name, tmp_path)
    _, one = run({}, *ARGS)
    assert one["PEERS"] == "127.0.0.1"
    done, two = run(
        {
            "VONK_WORLD_SIZE": "2",
            "VONK_RANK": "1",
            "VONK_LOCAL_ADDR": "10.9.0.2",
            "VONK_MASTER_PORT": "29555",
            "VONK_PEER_ADDRS": "10.9.0.1,10.9.0.2",
            "IPS": "x",
        },
        *ARGS,
    )
    assert done.returncode == 0, done.stderr
    assert (two["PEERS"], two["NODE_IP"], two["CONTROL_PORT"]) == (
        "10.9.0.1,10.9.0.2",
        "10.9.0.2",
        "29555",
    )
    assert "IPS" not in two
    refused, _ = run({"VONK_WORLD_SIZE": "2"}, *ARGS)
    assert refused.returncode != 0
