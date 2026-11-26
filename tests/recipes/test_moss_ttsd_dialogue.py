"""A MOSS-TTSD job's speaker count is decided from its slots before any model loads."""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest
from repo_root import ROOT

PIPELINES = ROOT / "adapters/audio/moss-ttsd-pytorch/pipelines"


def _dialogue() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "moss_ttsd_dialogue", PIPELINES / "dialogue.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_speaker_count_is_the_highest_tag_or_the_number_of_clips() -> None:
    speakers = _dialogue().speakers
    assert speakers("[S1] Hi. [S2] Hello. [S3] Hey.", 0, 0) == 3
    assert speakers("[S1] Hi there.", 2, 2) == 2
    # The creator's normalization reads a bare [2] as [S2].
    assert speakers("[1] Hi. [2] Hello.", 0, 0) == 2


@pytest.mark.parametrize(
    ("script", "clips", "transcripts"),
    [
        ("Hello without any speaker tag.", 0, 0),
        ("[S6] One speaker too many.", 0, 0),
        ("[S1] Hi.", 6, 6),
        ("[S1] Hi.", 2, 1),
    ],
)
def test_a_script_or_clip_set_the_model_cannot_run_is_refused(
    script: str, clips: int, transcripts: int
) -> None:
    with pytest.raises(SystemExit):
        _dialogue().speakers(script, clips, transcripts)
