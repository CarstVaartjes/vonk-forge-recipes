"""Check native D3 registration and the HTTP contract in the pinned image."""

from vllm.entrypoints.generate.structured_decisions.api_router import router
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
)
from vllm.model_executor.models import ModelRegistry

assert "D3ForSequenceClassification" in ModelRegistry.get_supported_archs()
assert any(route.path == "/v1/systemone" for route in router.routes)
request = StructuredDecisionRequest(
    state={"input": None},
    questions={
        "kind": {"type": "choice", "criteria": {"code": "Code", "chat": "Chat"}}
    },
)
assert request.state["input"] is None
