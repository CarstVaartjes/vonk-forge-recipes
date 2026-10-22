"""Restored playbook images must install and import the actual audio closure."""

from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("broken", [None, "librosa", "soundfile"])
def test_restored_image_rejects_missing_audio(dtype: str, broken: str | None) -> None:
    adapter = (
        ROOT
        / "adapters/nvidia"
        / (f"nemotron-3-nano-omni-30b-a3b-{dtype}-nvidia-vllm-single")
    )

    def import_audio(name: str) -> object:
        if name == broken:
            raise ImportError(f"missing {name}")
        return object()

    with (
        patch("os.access", return_value=True),
        patch("shutil.which", return_value="/usr/local/bin/vllm"),
        patch("importlib.import_module", side_effect=import_audio),
    ):
        if broken:
            with pytest.raises(ImportError, match=broken):
                runpy.run_path(str(adapter / "verify-runtime.py"))
        else:
            runpy.run_path(str(adapter / "verify-runtime.py"))


@pytest.mark.parametrize("dtype", ["bf16"])
def test_audio_install_uses_the_locked_missing_dependency(dtype: str) -> None:
    adapter = (
        ROOT
        / "adapters/nvidia"
        / (f"nemotron-3-nano-omni-30b-a3b-{dtype}-nvidia-vllm-single")
    )
    dockerfile = (adapter / "Dockerfile").read_text()
    assert "--require-hashes -r /opt/vonk/requirements-audio.lock" in dockerfile
    lock = (adapter / "requirements-audio.lock").read_text()
    assert "librosa==0.11.0" in lock
    assert "soundfile==0.14.0" in lock
