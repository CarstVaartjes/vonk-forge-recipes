#!/bin/sh
# Controller launcher for the one-Spark GLM-5.3-Flash SAGE-REXL3 lane: the engine arguments are the recipe's.
# The pack is mounted read-only at /models/target. vllm-exl3 needs a config.json that declares the pack's per-layer
# expert K and its dense EXL3 linears (the creator served a local "overlay" directory of that kind; his scripts do not
# build it), so make_overlay.py derives the same declaration from the pack's tensor headers into a directory under the
# writable cache: symlinks to the pack's shards, the new config.json, one extra safetensors file with the split q/k/v
# conv1d and projection tensors vLLM's glm5next loads (about 2.2 GiB, rebuilt at every start) and the index naming them.
# The creator's own self-test of the lookups his overlay must satisfy runs against it before vLLM starts. The recipe's model path is replaced by the overlay.
set -eu
PACK=/models/target
OVERLAY="${VLLM_EXL3_MODEL_DIR:?VLLM_EXL3_MODEL_DIR names the overlay directory}"
case "$OVERLAY" in
  /outputs/cache/*) ;;
  *) echo "VLLM_EXL3_MODEL_DIR must be under /outputs/cache, got $OVERLAY" >&2; exit 2 ;;
esac
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" /outputs/cache/mem
python3 /opt/vonk/make_overlay.py "$PACK" "$OVERLAY"
python3 /opt/vonk-selftest.py
count=$#
while [ "$count" -gt 0 ]; do
  argument=$1
  shift
  if [ "$argument" = "$PACK" ]; then argument=$OVERLAY; fi
  set -- "$@" "$argument"
  count=$((count - 1))
done
exec vllm "$@"
