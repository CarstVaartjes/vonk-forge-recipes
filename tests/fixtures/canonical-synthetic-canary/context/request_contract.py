"""Typed text-chat subset used by the deterministic HTTP test service."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None


class ChatRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    model: Annotated[str, Field(min_length=1)]
    messages: Annotated[list[ChatMessage], Field(min_length=1)]
    stream: bool | None = False
    max_tokens: Annotated[int, Field(gt=0)] | None = None
