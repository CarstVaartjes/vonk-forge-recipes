"""The name a running recipe is served under: the platform's endpoint alias.

A profile assignment's name, the client-facing model name and a reviewed
qualification service alias are all this one shape. It is owned here so the
recipe library, the sweep tooling and the Controller cannot drift apart: a
lowercase identifier a client can send as the OpenAI ``model`` and a profile
can key on, without case-insensitive collisions.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StrictStr

ENDPOINT_ALIAS_PATTERN = r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$"
ENDPOINT_ALIAS_MAX_LENGTH = 128

EndpointAlias = Annotated[
    StrictStr,
    Field(
        min_length=1,
        max_length=ENDPOINT_ALIAS_MAX_LENGTH,
        pattern=ENDPOINT_ALIAS_PATTERN,
    ),
]
