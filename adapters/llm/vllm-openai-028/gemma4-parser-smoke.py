#!/usr/bin/env python3
"""Build-time regression screen for vLLM's Gemma 4 tool parser."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.gemma4 import TOOL_CALL_END, TOOL_CALL_START
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser


START_ID = 48
END_ID = 49


def parser() -> Gemma4EngineToolParser:
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {
        TOOL_CALL_START: START_ID,
        TOOL_CALL_END: END_ID,
    }
    tokenizer.decode.side_effect = lambda ids: {
        START_ID: TOOL_CALL_START,
        END_ID: TOOL_CALL_END,
    }.get(ids[0], f"tok{ids[0]}")
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_status",
            "parameters": {
                "type": "object",
                "properties": {
                    "active": {"type": "boolean"},
                    "count": {"type": "integer"},
                },
            },
        },
    )
    return Gemma4EngineToolParser(tokenizer, tools=[tool])


def request() -> MagicMock:
    value = MagicMock(spec=ChatCompletionRequest)
    value.tools = []
    value.tool_choice = "auto"
    return value


def non_streaming() -> None:
    output = (
        "Checking. <|tool_call>call:set_status{active:true,count:42}<tool_call|>"
    )
    result = parser().extract_tool_calls(output, request())
    assert result.tools_called is True
    assert result.content == "Checking."
    assert result.tool_calls[0].function.name == "set_status"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "active": True,
        "count": 42,
    }


def streaming() -> None:
    instance = parser()
    chunks = [
        "<|tool_call>",
        "call:set_status{active:tru",
        "e,count:4",
        "2}",
        "<tool_call|>",
    ]
    previous_text = ""
    previous_ids: list[int] = []
    arguments = ""
    name = None
    for chunk in chunks:
        ids = []
        if TOOL_CALL_START in chunk:
            ids.append(START_ID)
        if TOOL_CALL_END in chunk:
            ids.append(END_ID)
        if not ids:
            ids.append(0)
        current_text = previous_text + chunk
        current_ids = previous_ids + ids
        delta = instance.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=tuple(previous_ids),
            current_token_ids=tuple(current_ids),
            delta_token_ids=tuple(ids),
            request=request(),
        )
        if delta is not None and delta.tool_calls:
            for call in delta.tool_calls:
                if call.function.name:
                    name = call.function.name
                arguments += call.function.arguments or ""
        previous_text = current_text
        previous_ids = current_ids
    assert name == "set_status"
    assert json.loads(arguments) == {"active": True, "count": 42}


if __name__ == "__main__":
    non_streaming()
    streaming()
