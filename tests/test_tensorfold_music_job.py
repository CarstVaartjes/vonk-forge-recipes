"""Declared audio inputs reach the native engine; corrupt or extra inputs never do."""

import hashlib
import importlib
import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "adapters/audio/drowzeys-music3-tensorfold"
sys.path.insert(0, str(next(CONTEXT.glob("*.whl"))))
Manifest = importlib.import_module(
    "vonk_agent_protocol.job_inputs"
).RecipeJobInputManifest
SPEC = importlib.util.spec_from_file_location("music3_job", CONTEXT / "job.py")
assert SPEC and SPEC.loader
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    data = b"Warm acoustic folk, gentle guitar."
    (root / "CAPTION.TXT").write_bytes(data)
    manifest = Manifest.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "total_bytes": len(data),
                "files": [
                    {
                        "slot": "caption",
                        "name": "CAPTION.TXT",
                        "media_type": "text/plain",
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
        )
    )
    (root / "manifest.json").write_text(manifest.model_dump_json())
    return root


def test_manifest_drives_engine_and_wav_is_checked(inputs, tmp_path, monkeypatch):
    calls = []

    def engine(command, check):
        assert check
        calls.append(command)
        with wave.open(command[command.index("--out") + 1], "wb") as audio:
            audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0" * 44100 * 4)

    monkeypatch.setattr(JOB.subprocess, "run", engine)
    JOB.run(inputs, tmp_path / "outputs", 60, 0)
    assert calls[0][calls[0].index("--caption-file") + 1] == str(inputs / "CAPTION.TXT")
    assert "--lyrics-file" not in calls[0]
    assert calls[0][calls[0].index("--model") + 1] == "/models/music3"
    (inputs / "extra.txt").write_text("unreviewed")
    with pytest.raises(ValueError, match="undeclared"):
        JOB.run(inputs, tmp_path / "outputs", 60, 0)
    assert len(calls) == 1


def test_changed_input_and_unbounded_duration_rejected(inputs, tmp_path):
    with pytest.raises(ValueError, match="duration"):
        JOB.run(inputs, tmp_path / "outputs", float("nan"), 0)
    path = inputs / "CAPTION.TXT"
    path.write_bytes(b"X" * path.stat().st_size)
    with pytest.raises(ValueError, match="digest"):
        JOB.inputs(inputs)
