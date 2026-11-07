"""The closed vocabulary of Model capabilities, and what each one means.

A Model display lists the capabilities its checkpoint supports, by name. The
names are a closed set: a name this release does not list is refused, so a new
capability is a contract minor release that readers adopt before a catalog
declares it. Each name has exactly one ``CapabilityRole``; readers and guards ask
the role instead of parsing the name.
"""

from __future__ import annotations

from enum import StrEnum


class CapabilityRole(StrEnum):
    # The Model reads a non-text input modality (image, audio or video).
    UNDERSTANDING = "understanding"
    # The Model produces or edits something other than text or code.
    GENERATOR = "generator"
    # Everything else: text and code output, serving features, classifiers.
    OTHER = "other"


class CapabilityName(StrEnum):
    THREE_D_GENERATION = "3d-generation"
    AUDIO_GENERATION = "audio-generation"
    AUDIO_UNDERSTANDING = "audio-understanding"
    CHAT = "chat"
    CODE_GENERATION = "code-generation"
    DECISION = "decision"
    IMAGE_EDITING = "image-editing"
    IMAGE_GENERATION = "image-generation"
    IMAGE_UNDERSTANDING = "image-understanding"
    IMAGE_UPSCALING = "image-upscaling"
    MATERIAL_ESTIMATION = "material-estimation"
    MOTION_CAPTURE = "motion-capture"
    MOTION_GENERATION = "motion-generation"
    OCR = "ocr"
    PARAMETRIC_BODY_MODEL = "parametric-body-model"
    PERSON_DETECTION = "person-detection"
    REASONING = "reasoning"
    SPECULATIVE_DECODING = "speculative-decoding"
    SPEECH_RECOGNITION = "speech-recognition"
    TEXT_GENERATION = "text-generation"
    TEXT_TO_SPEECH = "text-to-speech"
    TOOL_USE = "tool-use"
    VIDEO_GENERATION = "video-generation"
    VIDEO_UNDERSTANDING = "video-understanding"
    VOICE_CLONING = "voice-cloning"
    VOICE_DESIGN = "voice-design"

    @property
    def role(self) -> CapabilityRole:
        return CAPABILITY_ROLES[self]


_UNDERSTANDING = (
    CapabilityName.AUDIO_UNDERSTANDING,
    CapabilityName.IMAGE_UNDERSTANDING,
    CapabilityName.VIDEO_UNDERSTANDING,
)
_GENERATORS = (
    CapabilityName.THREE_D_GENERATION,
    CapabilityName.AUDIO_GENERATION,
    CapabilityName.IMAGE_EDITING,
    CapabilityName.IMAGE_GENERATION,
    CapabilityName.IMAGE_UPSCALING,
    CapabilityName.MATERIAL_ESTIMATION,
    CapabilityName.MOTION_CAPTURE,
    CapabilityName.MOTION_GENERATION,
    CapabilityName.PARAMETRIC_BODY_MODEL,
    CapabilityName.TEXT_TO_SPEECH,
    CapabilityName.VIDEO_GENERATION,
    CapabilityName.VOICE_CLONING,
    CapabilityName.VOICE_DESIGN,
)
_OTHER = (
    CapabilityName.CHAT,
    CapabilityName.CODE_GENERATION,
    CapabilityName.DECISION,
    CapabilityName.OCR,
    CapabilityName.PERSON_DETECTION,
    CapabilityName.REASONING,
    CapabilityName.SPECULATIVE_DECODING,
    CapabilityName.SPEECH_RECOGNITION,
    CapabilityName.TEXT_GENERATION,
    CapabilityName.TOOL_USE,
)

# The one classification; every name has an explicit entry.
CAPABILITY_ROLES: dict[CapabilityName, CapabilityRole] = {
    **dict.fromkeys(_UNDERSTANDING, CapabilityRole.UNDERSTANDING),
    **dict.fromkeys(_GENERATORS, CapabilityRole.GENERATOR),
    **dict.fromkeys(_OTHER, CapabilityRole.OTHER),
}
