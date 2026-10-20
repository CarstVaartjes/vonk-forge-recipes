"""Typed LiteLLM authentication evidence."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SmokeAuthClass(StrEnum):
    CREDENTIAL = "client-credential"
    REFUSAL = "client-authorization-refusal"


class GatewayAuthType(StrEnum):
    EXPIRED = "expired_key"
    MISSING = "token_not_found_in_db"


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
