"""Verify native D3 model and HTTP endpoint support in the actual pinned image."""
from vllm.model_executor.models import ModelRegistry
from vllm.entrypoints.generate.structured_decisions.api_router import router
from vllm.entrypoints.generate.structured_decisions.protocol import StructuredDecisionRequest
assert ModelRegistry.is_model_supported("D3ForSequenceClassification")
assert any(route.path == "/v1/systemone" for route in router.routes)
request = StructuredDecisionRequest(state={"input": None}, questions={"kind": {"type": "choice", "criteria": {"code": "Code", "chat": "Chat"}}})
assert request.state["input"] is None
