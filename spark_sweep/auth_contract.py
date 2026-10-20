"""Typed LiteLLM authentication evidence and durable renewal bounds."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SmokeAuthClass(StrEnum):
    CREDENTIAL = "client-credential"
    REFUSAL = "client-authorization-refusal"


class GatewayAuthType(StrEnum):
    EXPIRED = "expired_key"
    MISSING = "token_not_found_in_db"


class SmokeWaitReason(StrEnum):
    CLIENT_KEY = "client key renewal"


class GatewayError(BaseModel):
    model_config = ConfigDict(strict=True)

    type: str
    message: str


class GatewayAuthReply(BaseModel):
    model_config = ConfigDict(strict=True)

    error: GatewayError | None = None
    detail: str | None = None

    @property
    def renewable(self) -> bool:
        # A route denial takes precedence even if a wrapper supplies an auth type.
        if self.detail is not None:
            return False
        return self.error is not None and self.error.type in set(GatewayAuthType)


class ClientKeyWait(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: float
    deadline_at: float
