#!/usr/bin/env python3
"""Prevent terminal Inkling control tokens from leaking into API content."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: inkling-agent-parser-fix v2"

ANCHOR = """        return reasoning, content, tool_call_info

    @staticmethod
    def _extract_args_value(parsed: dict) -> str | None:
"""

PATCHED = """        # The unified parser in the pinned vLLM build can emit the dedicated
        # END_MESSAGE terminal as visible content after a tool result. Strip
        # only a terminal marker; it must not become an EOS token because it
        # also separates Inkling's reasoning and final-answer blocks.
        if content:
            content = content.removesuffix(END_MESSAGE) or None
        if tool_call_info.content:
            normalized_tool_content = tool_call_info.content.removesuffix(
                END_MESSAGE
            ) or None
            if normalized_tool_content != tool_call_info.content:
                tool_call_info = ExtractedToolCallInformation(
                    tools_called=tool_call_info.tools_called,
                    tool_calls=tool_call_info.tool_calls,
                    content=normalized_tool_content,
                )

        return reasoning, content, tool_call_info

    @staticmethod
    def _strip_terminal_content(content):
        if not content:
            return content
        return content.removesuffix(END_MESSAGE) or None

    def extract_reasoning(self, *args, **kwargs):
        reasoning, content = super().extract_reasoning(*args, **kwargs)
        return reasoning, self._strip_terminal_content(content)

    @staticmethod
    def _strip_terminal_content_delta(delta):
        if delta is None or not getattr(delta, "content", None):
            return delta
        normalized = delta.content.removesuffix(END_MESSAGE)
        if normalized == delta.content:
            return delta
        delta.content = normalized or None
        if not any(
            getattr(delta, field, None)
            for field in ("content", "reasoning", "tool_calls")
        ):
            return None
        return delta

    def extract_reasoning_streaming(self, *args, **kwargs):
        delta = super().extract_reasoning_streaming(*args, **kwargs)
        return self._strip_terminal_content_delta(delta)

    def finish_streaming(self):
        delta = super().finish_streaming()
        return self._strip_terminal_content_delta(delta)

    {marker}

    @staticmethod
    def _extract_args_value(parsed: dict) -> str | None:
""".format(marker=MARKER)


def patched_text(text: str) -> str:
    ast.parse(text)
    if MARKER in text:
        if "def extract_reasoning(self, *args, **kwargs):" not in text:
            raise ValueError("mod marker exists but non-streaming fix is missing")
        if "_strip_terminal_content_delta" not in text:
            raise ValueError("mod marker exists but streaming fix is missing")
        return text
    if "# spark-vllm mod: inkling-agent-parser-fix v1" in text:
        raise ValueError("v1 patch detected; recreate the container before applying v2")
    if text.count(ANCHOR) != 1:
        raise ValueError(
            f"expected exactly one Inkling return anchor; found {text.count(ANCHOR)}"
        )
    result = text.replace(ANCHOR, PATCHED, 1)
    ast.parse(result)
    compile(result, "<inkling-agent-parser-fix>", "exec")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        print(f"[inkling-agent-parser-fix ERROR] missing: {args.target}", file=sys.stderr)
        return 1
    original = args.target.read_text()
    try:
        patched = patched_text(original)
    except (SyntaxError, ValueError) as exc:
        print(f"[inkling-agent-parser-fix ERROR] {exc}", file=sys.stderr)
        return 1
    if args.check:
        state = "already patched" if patched == original else "compatible"
        print(f"[inkling-agent-parser-fix] {args.target} is {state}.")
        return 0
    if patched == original:
        print("[inkling-agent-parser-fix] already patched; skipping.")
        return 0
    temporary = args.target.with_suffix(args.target.suffix + ".agent-fix.tmp")
    temporary.write_text(patched)
    temporary.replace(args.target)
    print(f"[inkling-agent-parser-fix] patched {args.target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
