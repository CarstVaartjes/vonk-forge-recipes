"""Fail the image build unless vLLM exposes LFM2.5-VL's exact API contract."""

from vllm.model_executor.models.lfm2_vl import Lfm2VLForConditionalGeneration
from vllm.tool_parsers.lfm2_tool_parser import Lfm2ToolParser

assert "SupportsMultiModal" in {
    base.__name__ for base in Lfm2VLForConditionalGeneration.__mro__
}
assert Lfm2VLForConditionalGeneration.get_placeholder_str("image", 0) == "<image>"
assert Lfm2ToolParser.__name__ == "Lfm2ToolParser"
