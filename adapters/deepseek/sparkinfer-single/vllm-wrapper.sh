#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  printf 'sparkinfer-single: %s\n' "$*" >&2
  exit 2
}

require_value() {
  (($# >= 2)) || die "$1 requires a value"
}

[[ ${1:-} == serve ]] || die "expected the vLLM serve subcommand"
[[ $# -ge 2 ]] || die "expected the hydrated model path"
model_source=$2
shift 2

served_model_name=
host=
port=
max_model_len=
max_num_seqs=
max_num_batched_tokens=
max_cudagraph_capture_size=
gpu_memory_utilization=
kv_cache_dtype=
prefix_cache=0

while (($#)); do
  case "$1" in
    --served-model-name)
      require_value "$@"; served_model_name=$2; shift 2 ;;
    --host)
      require_value "$@"; host=$2; shift 2 ;;
    --port)
      require_value "$@"; port=$2; shift 2 ;;
    --max-model-len)
      require_value "$@"; max_model_len=$2; shift 2 ;;
    --max-num-seqs)
      require_value "$@"; max_num_seqs=$2; shift 2 ;;
    --max-num-batched-tokens)
      require_value "$@"; max_num_batched_tokens=$2; shift 2 ;;
    --max-cudagraph-capture-size)
      require_value "$@"; max_cudagraph_capture_size=$2; shift 2 ;;
    --gpu-memory-utilization)
      require_value "$@"; gpu_memory_utilization=$2; shift 2 ;;
    --kv-cache-dtype)
      require_value "$@"; kv_cache_dtype=$2; shift 2 ;;
    --enable-prefix-caching)
      prefix_cache=1; shift ;;
    *)
      die "unsupported harness argument: $1" ;;
  esac
done

[[ $(uname -m) == aarch64 ]] || die "the pinned image requires Linux aarch64"
[[ -d ${model_source} ]] || die "hydrated model directory is missing: ${model_source}"
[[ -r ${model_source}/config.json ]] || die "hydrated config.json is missing"
[[ -r ${model_source}/quantization_config.json ]] || die "hydrated quantization_config.json is missing"
[[ -r ${model_source}/model.safetensors.index.json ]] || die "hydrated weight index is missing"
[[ -r ${model_source}/EXL3_MANIFEST.json ]] || die "hydrated EXL3 manifest is missing"
[[ -r ${model_source}/REAP_K216_PLAN.json ]] || die "hydrated REAP keep plan is missing"

[[ ${served_model_name} == deepseek-v4-flash-0731-spark ]] || die "unexpected served model name"
[[ ${host} == 0.0.0.0 ]] || die "the endpoint must bind to 0.0.0.0"
[[ ${port} == 8000 ]] || die "the catalog interface requires port 8000"
[[ ${max_model_len} == 262144 ]] || die "the validated model limit is 262144"
[[ ${max_num_seqs} == 4 ]] || die "the validated concurrency is four sequences"
[[ ${max_num_batched_tokens} == 8224 ]] || die "the validated batch-token limit is 8224"
[[ ${max_cudagraph_capture_size} == 6 ]] || die "the validated CUDA graph width is six"
[[ ${gpu_memory_utilization} == 0.9465 ]] || die "the validated memory utilization is 0.9465"
[[ ${kv_cache_dtype} == nvfp4_ds_mla ]] || die "the validated sparse-MLA cache selector is required"
[[ ${prefix_cache} == 1 ]] || die "the validated profile requires prefix caching"

readonly executable_payload_revision=22f28d32b9b29b4352eaa380ff8c2c170b2847ab
readonly state_root=/outputs/${executable_payload_revision}
readonly model_dir=${state_root}/tp1
readonly draft_dir=${state_root}/dspark-draft-k64
readonly cache_dir=${state_root}/cache

umask 077
mkdir -p "${state_root}" "${cache_dir}"

if [[ ! -f ${model_dir}/rank-sliced-tp1-manifest.json ]]; then
  /opt/runtime-venv/bin/python \
    /opt/recipe/scripts/coalesce_rank_sliced_exl3.py \
    --input-dir "${model_source}" \
    --output-dir "${model_dir}" \
    --reuse-complete \
    --workers 1
fi

/opt/runtime-venv/bin/python /opt/recipe/scripts/verify_tp1_manifest.py \
  "${model_dir}"

if [[ ! -f ${draft_dir}/model.safetensors.index.json ]]; then
  /opt/runtime-venv/bin/python /opt/recipe/scripts/build_dspark_draft.py \
    --source "${model_dir}" \
    --output "${draft_dir}" \
    --experts 64 \
    --structured-per-category 32
fi

/opt/runtime-venv/bin/python /opt/recipe/scripts/selftest.py

export ALLREDUCE_MODE=nccl
export BACKEND=b12x-a8
export CUDAGRAPH_CAPTURE_SIZES=6
export DCP_SIZE=1
export DRAFT_SAMPLE_METHOD=probabilistic
export DSPARK_CAPACITY=0
export DSPARK_DRAFT_ATTENTION_BACKEND=B12X_MLA_SPARSE
export DSPARK_DYNAMIC_DRAFT_DEPTH=0
export DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW=8
export DSPARK_TOKENS=5
export GPU_MEMORY_UTILIZATION=${gpu_memory_utilization}
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export HOST=${host}
export INDEXER_BACKEND=b12x
export KV_CACHE_DTYPE=${kv_cache_dtype}
export KV_FP8_ROPE=0
export LOAD_FORMAT=instanttensor
export MAX_CUDAGRAPH_CAPTURE_SIZE=${max_cudagraph_capture_size}
export MAX_MODEL_LEN=${max_model_len}
export MAX_NUM_BATCHED_TOKENS=${max_num_batched_tokens}
export MAX_NUM_SEQS=${max_num_seqs}
export MODE=dspark
export MODEL_PATH=${model_dir}
export PORT=${port}
export PREFIX_CACHE=${prefix_cache}
export SERVED_MODEL_NAME=${served_model_name}
export SPEC_MODEL_PATH=${draft_dir}
export TP_SIZE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_DSV4_PADDED_NVFP4=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_B12X_WO_PROJECTION=1
export XDG_CACHE_HOME=${cache_dir}

exec /opt/vllm/serve-ds4-flash.sh
