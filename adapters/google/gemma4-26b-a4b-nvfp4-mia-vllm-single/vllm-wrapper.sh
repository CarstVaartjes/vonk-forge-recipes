#!/bin/sh
set -eu

fail() {
  printf 'gemma4-26b-a4b-nvfp4-mia-vllm-single: %s\n' "$*" >&2
  exit 2
}

[ "$(uname -m)" = aarch64 ] || fail "the pinned runtime requires Linux aarch64"
[ "${1:-}" = serve ] || fail "expected the vLLM serve subcommand"
[ "${2:-}" = /models/target ] || fail "expected the target model at /models/target"
[ -r /models/target/config.json ] || fail "missing hydrated target config.json"
[ -r /models/target/model.safetensors.index.json ] || fail "missing hydrated target weight index"
[ -r /models/assistant/model.safetensors ] || fail "missing hydrated MTP assistant weights"

for candidate in /opt/vllm/.venv/bin/vllm /usr/local/bin/vllm; do
  if [ -x "$candidate" ]; then
    exec "$candidate" "$@"
  fi
done
fail "vllm executable not found"
