#!/usr/bin/env bash
# The two sed rewrites of scripts/serve-hy3-tools.sh (upstream bug #3), applied at build time.
set -euo pipefail
RP=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/hy_v3_reasoning_parser.py
TP=/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/hy_v3_tool_parser.py
sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "$RP"
sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; s|"</tool_calls>"|"</tool_calls:opensource>"|g; s|"<tool_call>"|"<tool_call:opensource>"|g; s|"</tool_call>"|"</tool_call:opensource>"|g; s|"<tool_sep>"|"<tool_sep:opensource>"|g; s|"<arg_key>"|"<arg_key:opensource>"|g; s|"</arg_key>"|"</arg_key:opensource>"|g; s|"<arg_value>"|"<arg_value:opensource>"|g; s|"</arg_value>"|"</arg_value:opensource>"|g' "$TP"
grep -q "opensource" "$RP"
grep -q "opensource" "$TP"
