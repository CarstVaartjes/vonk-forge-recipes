#!/usr/bin/env bash
# Build-time form of the "bug #3 fix" that MiaAI-Lab's start.sh applies to the
# vLLM hy_v3 parsers at every serve: the checkpoint emits :opensource-suffixed
# special tokens while the parsers hardcode the bare forms. The sed commands are
# upstream's, unchanged; only the moment they run differs.
set -euo pipefail
RP=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/hy_v3_reasoning_parser.py
TP=/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/hy_v3_tool_parser.py
test -f "$RP" && test -f "$TP"
sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "$RP"
sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; s|"</tool_calls>"|"</tool_calls:opensource>"|g; s|"<tool_call>"|"<tool_call:opensource>"|g; s|"</tool_call>"|"</tool_call:opensource>"|g; s|"<tool_sep>"|"<tool_sep:opensource>"|g; s|"<arg_key>"|"<arg_key:opensource>"|g; s|"</arg_key>"|"</arg_key:opensource>"|g; s|"<arg_value>"|"<arg_value:opensource>"|g; s|"</arg_value>"|"</arg_value:opensource>"|g' "$TP"
echo "PATCH-CHECK reasoning:"; grep -c "opensource" "$RP"
echo "PATCH-CHECK tool:"; grep -c "opensource" "$TP"
