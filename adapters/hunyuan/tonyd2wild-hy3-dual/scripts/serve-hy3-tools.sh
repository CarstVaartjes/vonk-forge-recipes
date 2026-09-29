#!/usr/bin/env bash
exec > /tmp/hy3-serve-tools.log 2>&1
# bug #3 fix: this checkpoint emits :opensource-suffixed special tokens (HYTK=':opensource');
# vLLM's hy_v3 parsers hardcode the bare forms. Rewrite them to match ground truth.
RP=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/hy_v3_reasoning_parser.py
TP=/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/hy_v3_tool_parser.py
sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "$RP"
sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; s|"</tool_calls>"|"</tool_calls:opensource>"|g; s|"<tool_call>"|"<tool_call:opensource>"|g; s|"</tool_call>"|"</tool_call:opensource>"|g; s|"<tool_sep>"|"<tool_sep:opensource>"|g; s|"<arg_key>"|"<arg_key:opensource>"|g; s|"</arg_key>"|"</arg_key:opensource>"|g; s|"<arg_value>"|"<arg_value:opensource>"|g; s|"</arg_value>"|"</arg_value:opensource>"|g' "$TP"
echo "PATCH-CHECK reasoning:"; grep -c "opensource" "$RP"
echo "PATCH-CHECK tool:"; grep -c "opensource" "$TP"
export VLLM_HOST_IP=192.168.192.1
exec vllm serve /models \
  --served-model-name hy3 --port 8600 \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 131072 --max-num-seqs 6 \
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice \
  --trust-remote-code --enforce-eager
