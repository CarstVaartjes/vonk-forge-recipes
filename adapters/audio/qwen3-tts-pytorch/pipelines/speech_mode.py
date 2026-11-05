"""Which Qwen3-TTS model a job runs, decided once from the slots it was given."""

from __future__ import annotations

import enum
from collections.abc import Collection


class Mode(enum.Enum):
    DESIGN = "design"
    CLONE_WITH_TRANSCRIPT = "clone-with-transcript"
    CLONE_SPEAKER_ONLY = "clone-speaker-only"


def select_mode(slots: Collection[str]) -> Mode:
    """Map the optional input slots present to a mode, or refuse before any load.

    A reference clip selects cloning with the Base model (in-context when its
    transcript is given, speaker embedding only otherwise); a voice description
    selects VoiceDesign. A slot the chosen model cannot use is refused rather
    than ignored.
    """
    if "reference_audio" in slots:
        if "voice" in slots:
            raise SystemExit(
                "voice and reference_audio are exclusive: a voice description "
                "drives VoiceDesign and a reference clip drives cloning; give one"
            )
        if "reference_text" in slots:
            return Mode.CLONE_WITH_TRANSCRIPT
        return Mode.CLONE_SPEAKER_ONLY
    if "reference_text" in slots:
        raise SystemExit("reference_text is the transcript of reference_audio; add it")
    if "voice" in slots:
        return Mode.DESIGN
    raise SystemExit("give a voice description (voice) or a reference clip (reference_audio)")
