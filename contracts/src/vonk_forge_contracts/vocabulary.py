"""The words every consumer shows for capabilities and qualification statuses.

The contract owns the names (:class:`CapabilityName`, :class:`QualificationStatus`)
and, here, their user-facing text: a capability's label, description and
category, a status's meaning and whether it counts as verified on a Spark. The
release index publishes this vocabulary, so the site, the Controller UI and
vonkctl render the same words from the release they serve instead of keeping
hand copies. Every enum member must have an entry; a missing one fails at
import, so the vocabulary is complete by construction.

A status counts as verified only when a real hardware pass backs the current
content, which is ``Working``. Published names are plain strings, so a reader on
an older minor tolerates a name a newer minor adds.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from .capability import CapabilityName
from .qualification_status import QualificationStatus


class CapabilityCategory(StrEnum):
    LANGUAGE = "language"
    CODE = "code"
    VISION = "vision"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    THREE_D = "3d"
    OTHER = "other"


CATEGORY_LABEL: Mapping[CapabilityCategory, str] = {
    CapabilityCategory.LANGUAGE: "Language",
    CapabilityCategory.CODE: "Code",
    CapabilityCategory.VISION: "Vision",
    CapabilityCategory.IMAGE: "Image",
    CapabilityCategory.VIDEO: "Video",
    CapabilityCategory.AUDIO: "Audio",
    CapabilityCategory.THREE_D: "3D",
    CapabilityCategory.OTHER: "Other",
}


class CapabilityText(NamedTuple):
    label: str
    description: str
    category: CapabilityCategory


class StatusText(NamedTuple):
    meaning: str
    verified: bool


_C = CapabilityCategory
CAPABILITY_TEXT: Mapping[CapabilityName, CapabilityText] = {
    CapabilityName.THREE_D_GENERATION: CapabilityText(
        "3D generation",
        "Creates 3D assets such as meshes from text or images.",
        _C.THREE_D,
    ),
    CapabilityName.AUDIO_GENERATION: CapabilityText(
        "Audio generation", "Creates music or sound.", _C.AUDIO
    ),
    CapabilityName.AUDIO_UNDERSTANDING: CapabilityText(
        "Audio understanding", "Takes audio as input and answers about it.", _C.AUDIO
    ),
    CapabilityName.CHAT: CapabilityText(
        "Chat", "Holds a conversation in turns.", _C.LANGUAGE
    ),
    CapabilityName.CODE_GENERATION: CapabilityText(
        "Code generation", "Writes and edits program code.", _C.CODE
    ),
    CapabilityName.DECISION: CapabilityText(
        "Decision", "Chooses between options and returns the choice.", _C.OTHER
    ),
    CapabilityName.IMAGE_EDITING: CapabilityText(
        "Image editing", "Changes an existing image from an instruction.", _C.IMAGE
    ),
    CapabilityName.IMAGE_GENERATION: CapabilityText(
        "Image generation", "Creates images from text.", _C.IMAGE
    ),
    CapabilityName.IMAGE_UNDERSTANDING: CapabilityText(
        "Image understanding",
        "Takes images as input and answers about them.",
        _C.VISION,
    ),
    CapabilityName.IMAGE_UPSCALING: CapabilityText(
        "Image upscaling", "Raises the resolution of an image.", _C.IMAGE
    ),
    CapabilityName.MATERIAL_ESTIMATION: CapabilityText(
        "Material estimation",
        "Estimates surface materials of a 3D asset or image.",
        _C.THREE_D,
    ),
    CapabilityName.MOTION_CAPTURE: CapabilityText(
        "Motion capture", "Recovers body motion from video.", _C.THREE_D
    ),
    CapabilityName.MOTION_GENERATION: CapabilityText(
        "Motion generation", "Creates character motion from text.", _C.THREE_D
    ),
    CapabilityName.OCR: CapabilityText(
        "OCR", "Reads text from images and documents.", _C.VISION
    ),
    CapabilityName.PARAMETRIC_BODY_MODEL: CapabilityText(
        "Parametric body model",
        "Represents a human body by shape and pose parameters.",
        _C.THREE_D,
    ),
    CapabilityName.PERSON_DETECTION: CapabilityText(
        "Person detection", "Finds people in images or video.", _C.VISION
    ),
    CapabilityName.REASONING: CapabilityText(
        "Reasoning", "Works through a problem in steps before answering.", _C.LANGUAGE
    ),
    CapabilityName.SPECULATIVE_DECODING: CapabilityText(
        "Speculative decoding",
        "A draft model that speeds up another model's text generation.",
        _C.LANGUAGE,
    ),
    CapabilityName.SPEECH_RECOGNITION: CapabilityText(
        "Speech recognition", "Turns speech into text.", _C.AUDIO
    ),
    CapabilityName.TEXT_GENERATION: CapabilityText(
        "Text generation", "Continues or writes text from a prompt.", _C.LANGUAGE
    ),
    CapabilityName.TEXT_TO_SPEECH: CapabilityText(
        "Text to speech", "Speaks text aloud.", _C.AUDIO
    ),
    CapabilityName.TOOL_USE: CapabilityText(
        "Tool use", "Calls functions or tools you describe.", _C.LANGUAGE
    ),
    CapabilityName.VIDEO_GENERATION: CapabilityText(
        "Video generation", "Creates video clips.", _C.VIDEO
    ),
    CapabilityName.VIDEO_UNDERSTANDING: CapabilityText(
        "Video understanding", "Takes video as input and answers about it.", _C.VISION
    ),
    CapabilityName.VOICE_CLONING: CapabilityText(
        "Voice cloning", "Speaks in the voice of a reference recording.", _C.AUDIO
    ),
    CapabilityName.VOICE_DESIGN: CapabilityText(
        "Voice design", "Creates a new voice from a description.", _C.AUDIO
    ),
}

STATUS_TEXT: Mapping[QualificationStatus, StatusText] = {
    QualificationStatus.WORKING: StatusText(
        "Passed on a real DGX Spark for this exact recipe, runtime image and model.",
        True,
    ),
    QualificationStatus.RECHECK: StatusText(
        "Passed before, but the recipe, image or model has changed since.", False
    ),
    QualificationStatus.FIX_TO_VERIFY: StatusText(
        "A recipe failure was followed by a change to the recipe, image or model.",
        False,
    ),
    QualificationStatus.FAILED_RECIPE: StatusText(
        "Failed on a real DGX Spark with the current recipe, image and model.", False
    ),
    QualificationStatus.BLOCKED_PLATFORM: StatusText(
        "Could not run because of a platform fault, not because of the recipe.", False
    ),
    QualificationStatus.UNTESTED: StatusText("No hardware verdict yet.", False),
    QualificationStatus.CANNOT_TEST: StatusText(
        "Needs more Sparks than the test fleet has.", False
    ),
    QualificationStatus.TESTING: StatusText(
        "Being tested on hardware now; the earlier result stays visible.", False
    ),
    QualificationStatus.NOT_YET_VERIFIED: StatusText(
        "No hardware verdict yet; the reason is shown under the badge.", False
    ),
}


def require_complete(
    name: str, entries: Collection[object], members: Collection[object]
) -> None:
    """Raise unless ``entries`` covers exactly ``members``."""

    missing = sorted(str(m) for m in set(members) - set(entries))
    extra = sorted(str(e) for e in set(entries) - set(members))
    if missing or extra:
        raise RuntimeError(f"{name} vocabulary: missing {missing}, unknown {extra}")


require_complete("capability", CAPABILITY_TEXT, CapabilityName)
require_complete("status", STATUS_TEXT, QualificationStatus)
require_complete("category", CATEGORY_LABEL, CapabilityCategory)


class _VocabularyContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VocabularyCategory(_VocabularyContract):
    id: StrictStr
    label: StrictStr


class VocabularyCapability(_VocabularyContract):
    name: StrictStr
    label: StrictStr
    description: StrictStr
    category: StrictStr


class VocabularyStatus(_VocabularyContract):
    status: StrictStr
    meaning: StrictStr
    verified: StrictBool


class CatalogVocabulary(_VocabularyContract):
    """The words of one release, as the index publishes them."""

    categories: tuple[VocabularyCategory, ...]
    capabilities: tuple[VocabularyCapability, ...]
    statuses: tuple[VocabularyStatus, ...]


def catalog_vocabulary() -> CatalogVocabulary:
    """The vocabulary this contract version publishes."""

    return CatalogVocabulary(
        categories=tuple(
            VocabularyCategory(id=category.value, label=CATEGORY_LABEL[category])
            for category in CapabilityCategory
        ),
        capabilities=tuple(
            VocabularyCapability(
                name=name.value,
                label=text.label,
                description=text.description,
                category=text.category.value,
            )
            for name, text in CAPABILITY_TEXT.items()
        ),
        statuses=tuple(
            VocabularyStatus(
                status=status.value, meaning=text.meaning, verified=text.verified
            )
            for status, text in STATUS_TEXT.items()
        ),
    )
