"""The gr8 Inkling-2 placement wrapper maps the recipe's arguments onto the creator's serve.sh."""

import os
import subprocess

import pytest
from repo_root import ROOT

ADAPTER = ROOT / "adapters/inkling/kindlingai-gr8-ink2-single"
FAKE_SERVE = """#!/bin/bash
env > "$DUMP"
"""


@pytest.fixture
def run(tmp_path):
    """The wrapper with the container's absolute paths moved under tmp_path."""
    script = (ADAPTER / "gr8-serve").read_text()
    (tmp_path / "recipe").mkdir()
    (tmp_path / "recipe/serve.sh").write_text(FAKE_SERVE)
    for old in ("/recipe", "/outputs", "/models"):
        script = script.replace(old, f"{tmp_path}{old}")
    wrapper = tmp_path / "gr8-serve"
    wrapper.write_text(script)
    wrapper.chmod(0o755)

    def invoke(*args, env=None):
        full = {
            "PATH": os.environ["PATH"],
            "DUMP": str(tmp_path / "dump"),
            **(env or {}),
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

    return tmp_path, invoke


def test_arguments_become_the_variables_serve_sh_reads(run):
    _, invoke = run
    done, dump = invoke(
        "--host",
        "0.0.0.0",
        "--port=8100",
        "--max-model-len",
        "262144",
        "--sess=4",
    )
    assert done.returncode == 0, done.stderr
    assert (dump["HOST"], dump["PORT"], dump["CAP"], dump["SESSIONS"]) == (
        "0.0.0.0",
        "8100",
        "262144",
        "4",
    )


def test_model_in_place_and_state_under_outputs(run):
    root, invoke = run
    done, dump = invoke("--port", "8100")
    assert done.returncode == 0, done.stderr
    assert dump["MODEL_DIR"] == f"{root}/models/target"
    assert dump["WORK_DIR"] == f"{root}/outputs/cache/gr8/ink2"
    assert dump["TMPDIR"] == f"{root}/outputs/tmp"
    assert (root / "outputs/tmp").is_dir()


def test_unknown_argument_is_refused(run):
    _, invoke = run
    done, dump = invoke("--tp", "2")
    assert done.returncode == 2
    assert dump == {}
