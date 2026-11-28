"""Every closed set a reader keeps unknown words of (``KeepUnrecognised``).

The contract owns these words. A producer refuses a word this release does not
list; a consumer reader keeps it as an ``UnrecognisedWord`` and does not act on
it, so a newer minor may add a member. A new capability is a member of
:class:`CapabilityName`, its entry in one of the role tuples (``_UNDERSTANDING``,
``_GENERATORS``, ``_OTHER``) and a ``CAPABILITY_TEXT`` row. The
sets the platform executes live in :mod:`vonk_forge_contracts.executed`.
``tools/check-closed-sets`` keeps every closed set in one of the two modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


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
    BACKGROUND_REMOVAL = "background-removal"
    CHAT = "chat"
    CODE_GENERATION = "code-generation"
    DECISION = "decision"
    EMBEDDING = "embedding"
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
    RETOPOLOGY = "retopology"
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
    CapabilityName.BACKGROUND_REMOVAL,
    CapabilityName.IMAGE_EDITING,
    CapabilityName.IMAGE_GENERATION,
    CapabilityName.IMAGE_UPSCALING,
    CapabilityName.MATERIAL_ESTIMATION,
    CapabilityName.MOTION_CAPTURE,
    CapabilityName.MOTION_GENERATION,
    CapabilityName.PARAMETRIC_BODY_MODEL,
    CapabilityName.RETOPOLOGY,
    CapabilityName.TEXT_TO_SPEECH,
    CapabilityName.VIDEO_GENERATION,
    CapabilityName.VOICE_CLONING,
    CapabilityName.VOICE_DESIGN,
)
_OTHER = (
    CapabilityName.CHAT,
    CapabilityName.CODE_GENERATION,
    CapabilityName.DECISION,
    CapabilityName.EMBEDDING,
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


class CapabilityCategory(StrEnum):
    LANGUAGE = "language"
    CODE = "code"
    VISION = "vision"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    THREE_D = "3d"
    OTHER = "other"


Modality = Literal["text", "image", "audio", "video", "3d", "embeddings"]


#: How a Model's weights relate to its release: the vendor's own released
#: checkpoint (``base``), the same weights in another format or bit-width
#: (``quantization``), or different weights such as a prune, distill or tune
#: (``derivative``).
VariantKind = Literal["base", "quantization", "derivative"]


Alignment = Literal[
    "standard", "abliterated", "derisked", "other-modified", "unspecified"
]


Endpoint = Literal[
    "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages"
]


# The upload formats of OpenAI's transcription endpoint.
AudioInputFormat = Literal[
    "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "ogg", "wav", "webm"
]


TranscriptionResponseFormat = Literal[
    "json", "text", "srt", "verbose_json", "vtt", "diarized_json"
]


# OpenAI createSpeech response formats.
SpeechResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


TimestampGranularity = Literal["word", "segment"]


# OpenAI createEmbedding encoding formats.
EmbeddingEncodingFormat = Literal["float", "base64"]


ToolChoice = Literal["auto", "none", "required", "named"]


ReasoningDefault = Literal["on", "off", "model"]


RequestPlacement = Literal["chat_template_kwargs", "top_level"]


class QualificationStatus(StrEnum):
    WORKING = "Working"
    RECHECK = "Re-check: worked before"
    FIX_TO_VERIFY = "Fix to verify"
    FAILED_RECIPE = "Failed: recipe"
    BLOCKED_PLATFORM = "Blocked: platform"
    UNTESTED = "Untested"
    CANNOT_TEST = "Can't test here"
    TESTING = "Testing now"
    NOT_YET_VERIFIED = "Not yet verified"


class QualificationOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class QualificationBlame(StrEnum):
    RECIPE = "recipe"
    PLATFORM = "platform"


class QualificationCauseCode(StrEnum):
    HARDWARE_PASS = "qualification.hardware_pass"
    RECIPE_FAILURE = "qualification.recipe_failure"
    PLATFORM_FAILURE = "qualification.platform_failure"
    RUNTIME_EXIT = "recipe.runtime_exit"
    RUNTIME_OOM = "recipe.runtime_oom"
    MODEL_INTEGRITY = "model.digest_mismatch"
    RECIPE_INTEGRITY = "recipe.digest_mismatch"
    INVALID_DEFINITION = "recipe.invalid_definition"
    INVALID_DOCUMENT = "recipe.invalid_document"
    INVALID_OPTION = "recipe.invalid_option"
    BUILD_POLICY = "dockerfile.heredoc_forbidden"
    SERVING_ASSERTION = "qualification.serving_assertion"
    JOB_FAILURE = "qualification.job_failure"


class QualificationSummary(StrEnum):
    PASSED = "Hardware checks passed"
    RECIPE_FAILED = "Recipe checks failed; needs fix"
    PLATFORM_FAILED = "Platform issue under fix"
    RUNTIME_EXIT = "Recipe runtime exited; needs fix"
    RUNTIME_OOM = "Recipe runtime exceeded available memory; needs fix"
    MODEL_INTEGRITY = "Model content integrity check failed; needs fix"
    INVALID_RECIPE = "Recipe definition or build policy check failed; needs fix"
    SERVING_ASSERTION = "Serving checks failed; needs fix"
    JOB_FAILURE = "Artifact job failed; needs fix"


class QualificationQualityNote(StrEnum):
    ANSWER_VARIANCE = "Model output differed from the expected answer"
    CARRIED = "Carried from the identical document before the display split"
    CARRIED_MODEL_FACTS = (
        "Carried from the identical execution document before the Model facts split"
    )
    # A document whose execution bytes did not change, re-digested by a contract
    # move (only its Model pins differ): its evidence is the old digest's.
    CARRIED_IDENTICAL_EXECUTION = "Carried from the identical execution document"


class QualificationChange(StrEnum):
    RECIPE = "recipe"
    IMAGE = "image"
    SOURCE = "source"
    MODEL = "model"


class QualificationUnverifiedReason(StrEnum):
    NEVER_TESTED = "never tested"
    NEEDS_SPARKS = "needs more Sparks"
    PLATFORM_ISSUE = "platform issue under fix"
    INCOMPLETE_PROVENANCE = "historical platform or runtime identity unavailable"


class QualificationHardware(StrEnum):
    GB10 = "GB10"


class QualificationTopology(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


PublishedStatus = Literal[
    QualificationStatus.WORKING,
    QualificationStatus.RECHECK,
    QualificationStatus.FIX_TO_VERIFY,
    QualificationStatus.FAILED_RECIPE,
    QualificationStatus.NOT_YET_VERIFIED,
]


# Which request field a context limit applies to.
ContextBasis = Literal["context", "slot_context"]
