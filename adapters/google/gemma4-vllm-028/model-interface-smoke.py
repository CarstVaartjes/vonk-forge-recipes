"""Fail the image build unless vLLM exposes Gemma 4's multimodal contract."""

from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

assert "SupportsMultiModal" in {
    base.__name__ for base in Gemma4ForConditionalGeneration.__mro__
}
