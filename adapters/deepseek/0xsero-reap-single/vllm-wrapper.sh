#!/usr/bin/env bash
# Vonk entrypoint for the 0xSero DeepSeek V4 Flash REAP single-Spark profiles.
# Upstream's launcher runs the GB10 patcher and then `vllm serve` inside the
# container; here the patcher is applied at image build time and this wrapper
# only starts the server with the recipe's argv.
set -Eeuo pipefail

[[ ${1:-} == serve ]] || { echo "deepseek-reap-single: expected the vLLM serve subcommand" >&2; exit 2; }
[[ $(uname -m) == aarch64 ]] || { echo "deepseek-reap-single: the pinned image requires Linux aarch64" >&2; exit 2; }
exec /usr/local/bin/vllm "$@"
