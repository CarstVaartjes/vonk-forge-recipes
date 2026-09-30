#!/usr/bin/env bash
# Vonk entrypoint for the 0xSero GLM-5.3-Flash EXL3 K2 + native MTP profile.
#
# Upstream's start.sh derives the native MTP draft view (symlinks plus a filtered
# weight index over the same checkpoint) on the host before `docker run`. The
# platform mounts the checkpoint read-only at /models, so the same upstream
# script runs here first, into the writable outputs volume, and the recipe's
# speculative config names that view. Everything else is the recipe's argv.
set -Eeuo pipefail

die() {
  printf 'glm53-exl3-k2-single: %s\n' "$*" >&2
  exit 2
}

[[ ${1:-} == serve ]] || die "expected the vLLM serve subcommand"
[[ $# -ge 2 ]] || die "expected the hydrated model path"
model_source=$2
shift 2

[[ $(uname -m) == aarch64 ]] || die "the pinned image requires Linux aarch64"
[[ -r ${model_source}/config.json ]] || die "hydrated config.json is missing"
[[ -r ${model_source}/model.safetensors.index.json ]] || die "hydrated weight index is missing"

readonly draft_dir=/outputs/mtp
rm -rf "${draft_dir}"
python3 /opt/vonk/tools/prepare_mtp_draft.py \
  --source "${model_source}" \
  --output "${draft_dir}" \
  --container-source "${model_source}"

exec python3 -m vllm.entrypoints.openai.api_server --model "${model_source}" "$@"
