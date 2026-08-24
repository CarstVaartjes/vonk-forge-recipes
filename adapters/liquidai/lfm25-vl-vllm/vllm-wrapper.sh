#!/bin/sh
set -eu

fail() {
  printf 'lfm25-vl-vllm: %s\n' "$*" >&2
  exit 2
}

[ "$(uname -m)" = aarch64 ] || fail "the pinned runtime requires Linux aarch64"
[ "${1:-}" = serve ] || fail "expected the vLLM serve subcommand"
[ "${2:-}" = /models ] || fail "expected the immutable model at /models"
[ -r /models/config.json ] || fail "missing hydrated config.json"
[ -r /models/model.safetensors ] || fail "missing hydrated model.safetensors"
[ -r /models/processor_config.json ] || fail "missing hydrated processor_config.json"
[ -r /models/tokenizer.json ] || fail "missing hydrated tokenizer.json"
[ -d /inputs ] || fail "missing read-only multimodal input mount"
[ -d /outputs ] || fail "missing writable output/cache mount"

exec vllm "$@"
