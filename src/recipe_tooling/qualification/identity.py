"""Execution identity: what ran, with the image named by its manifest digest.

The one owner of comparing qualification identities. A recipe image is
identified by its digest; ``runtime_image.build_key`` is only the recipes CI's
handle for finding a build, so no comparison reads it (an adopted image keeps its
digest under a new key, and its evidence still applies).
``tools/check-identity-compare`` rejects an added comparison anywhere else.
"""

from __future__ import annotations

from vonk_forge_contracts import QualificationIdentity


def image_digest(identity: QualificationIdentity) -> str | None:
    return (
        None if identity.runtime_image is None else identity.runtime_image.image_digest
    )


def same_execution(
    previous: QualificationIdentity, current: QualificationIdentity
) -> bool:
    """Whether a record's identity names the current execution.

    Recorded content is compared; missing historical provenance on the record
    (no source bundle, no image, no Spark count) never turns into a change.
    Publication separately requires fully resolved exact proof (``execution_key``).
    """

    return (
        previous.content_sha256 == current.content_sha256
        and (
            previous.source_bundle_sha256 is None
            or previous.source_bundle_sha256 == current.source_bundle_sha256
        )
        and previous.model_digests == current.model_digests
        and previous.models_resolved == current.models_resolved
        and previous.required_sparks in (None, current.required_sparks)
        and (
            previous.runtime_image is None
            or image_digest(previous) == image_digest(current)
        )
    )


def execution_key(identity: QualificationIdentity) -> str:
    """Every field of the identity, the image by its digest only: equal keys are exact."""

    return identity.model_dump_json(exclude={"runtime_image": {"build_key"}})
