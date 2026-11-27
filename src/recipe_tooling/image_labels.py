"""The upstream an adapter's Dockerfile labels pin (standard library only).

``org.opencontainers.image.source`` names a GitHub repository and
``org.opencontainers.image.revision`` a 40-hex commit. The vendored checker and
the added-line guards read the pin through this one rule.
"""

from __future__ import annotations

import re

SHA = re.compile(r"^[0-9a-f]{40}$")
GITHUB = re.compile(
    r"^https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
LABEL_SOURCE = re.compile(r'org\.opencontainers\.image\.source="([^"]+)"')
LABEL_REVISION = re.compile(r'org\.opencontainers\.image\.revision="([^"]+)"')


def label_source(dockerfile: str) -> tuple[str, str] | None:
    """The GitHub ``owner/name`` and commit the labels pin, or None."""
    source, revision = (
        LABEL_SOURCE.search(dockerfile),
        LABEL_REVISION.search(dockerfile),
    )
    if not source or not revision:
        return None
    match = GITHUB.match(source.group(1))
    if not match or not SHA.match(revision.group(1)):
        return None
    return match.group(1), revision.group(1)
