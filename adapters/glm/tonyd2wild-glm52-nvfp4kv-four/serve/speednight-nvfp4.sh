#!/usr/bin/env bash

# --- PRE-BOOT MEMORY RECLAIM (GB10 unified-mem: restore CUDA-free after churn/OOM) ---
# On GB10 CUDA-free tracks MemAvailable; OOM-kills/downloads leak orphaned GPU state
# + page cache + swap, depressing it ~1-2GB and failing razor-tuned gmu 0.91.
sudo fuser -k -9 /dev/nvidia* 2>/dev/null || true
sleep 2
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
sudo swapoff -a 2>/dev/null; sudo swapon -a 2>/dev/null || true
echo "[reclaim] Bluey CUDA-free restored: MemAvailable=$(grep MemAvailable /proc/meminfo | awk "{print \$2}")kB"
# --- END PRE-BOOT RECLAIM ---

#
# launch.sh — start GLM-5.2 (TP=4) across a 4-node GB10 / DGX Spark cluster.
#
# Self-contained: a plain `docker run` per node, no external harness. Multi-node
# is vLLM's NATIVE mechanism (--nnodes/--node-rank/--master-addr/--master-port);
# the workers (rank >= 1) start headless, then the head (rank 0) serves the API.
# There is NO Ray and NO shared-filesystem requirement — each node just needs the
# weights present locally (see WEIGHTS_DIR) and the kernels deployed (see step 3
# of bootstrap.sh, which writes them to KERNELS_DIR).
#
# Run from the HEAD node (NODES[0]). Workers are reached over key-based SSH.
#
#   ./launch.sh            # launch
#   ./launch.sh --dry-run  # print the docker commands without running them
#   ./launch.sh --stop     # docker rm -f the container on every node
#
# License: Apache-2.0.
set -uo pipefail

# ============================================================================
# CONFIG — edit these for your cluster
# ============================================================================
# RoCE rail IPs, rank 0 (head) FIRST. Run this script from the head node.
NODES=(192.168.192.1 192.168.192.2 192.168.192.3 192.168.192.4)
SSH_USER="dryrun-only"
SSH_KEY="/home/tonyspark1/.ssh/id_ed25519_shared"
node_user() { echo "tonyspark${1##*.}"; }

IMAGE="vllm-node-tf5-glm52-b12x:nvfp4-v1"
NAME="vllm_slot"                         # container name on every node
PORT=8210                                # OpenAI API port on the head node
MASTER_PORT=29501                        # vLLM cross-node rendezvous port

# Weights. A directory THAT EXISTS ON EVERY NODE, holding the HF hub layout:
#   $WEIGHTS_DIR/hub/glm52-int4-int8mix        (QuantTrio Int4-Int8Mix weights)
#   $WEIGHTS_DIR/hub/glm52-mtp-int4-aligned   (MTP draft)
#   $WEIGHTS_DIR/hub/nccl-2.30.4/libnccl.so.2 (LD_PRELOADed NCCL, see bootstrap step 2)
# Mounted writable at /cache/huggingface (HF_HOME) so JIT/Triton caches stay local.
# How the weights get onto each node is YOUR choice — per-node copy, rsync, or a
# shared mount pointed here. This script does not assume any of them.
WEIGHTS_DIR="/var/tmp/models"

# Per-node directory holding the Triton sparse-MLA kernels (bootstrap step 3
# deploys kernels/*.py here). Bound file-by-file over the vLLM tree, read-only.
KERNELS_DIR="/var/tmp/glm-triton-nvfp4"

# Per-node LOCAL writable compile/kernel cache (triton autotune, torch-inductor,
# vLLM cache). MUST live on local disk and NOT under WEIGHTS_DIR: for NFS-head
# serving WEIGHTS_DIR is a READ-ONLY NFS mount, so triton/inductor autotune
# writes into it fail with [Errno 30] Read-only file system, killing worker
# ranks during torch.compile. Bound in at /compile-cache; the three cache envs
# below point there. Safe for local-weights launches too (just a local dir).
COMPILE_CACHE_DIR="/var/tmp/glm-compile-cache"
# ============================================================================

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

DRYRUN=0; STOP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRYRUN=1 ;;
    --stop)    STOP=1 ;;
    *) die "unknown arg: $a (use --dry-run or --stop)" ;;
  esac
done

[ "${#NODES[@]}" -ge 1 ] || die "NODES is empty"
NNODES="${#NODES[@]}"
HEAD="${NODES[0]}"

if [ "$STOP" = 1 ]; then
  say "stopping '$NAME' on all ${NNODES} nodes"
  for ip in "${NODES[@]}"; do
    ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$(node_user "$ip")@$ip" "docker rm -f $NAME 2>/dev/null" \
      && printf '   stopped on %s\n' "$ip"
  done
  exit 0
fi

# ----------------------------------------------------------------------------
# Container env. The NCCL_IB_HCA /
# *_SOCKET_IFNAME values are RoCE-fabric-specific: set them for YOUR cluster
# (HCAs via `ibdev2netdev`, interfaces via `ip link`). Marked EDIT.
ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2"
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/compile-cache/triton"
  -e "TORCHINDUCTOR_CACHE_DIR=/compile-cache/inductor"
  -e "VLLM_CACHE_ROOT=/compile-cache/vllm"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256"
  -e "GLM52_BIND_HOST_TRITON=1"
  -e "GLM52_MQA_LOGITS_TRITON=1"
  -e "GLM52_PAGED_MQA_TRITON=1"
  -e "GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192"
  -e "GLM52_B12X_MLA=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=rocep1s0f0"
  -e "NCCL_SOCKET_IFNAME=enp1s0f0np0"
  -e "GLOO_SOCKET_IFNAME=enp1s0f0np0"
  -e "NCCL_IB_GID_INDEX=3"
  -e "NCCL_MAX_NCHANNELS=4"
  -e "NCCL_MIN_NCHANNELS=4"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "VLLM_MARLIN_USE_ATOMIC_ADD=1"
  -e "NCCL_DEBUG=WARN"
)

# Triton sparse-MLA kernels, bound read-only over the vLLM tree (matches
# GLM52_BIND_HOST_TRITON=1). Paths are inside the image's vLLM install.
MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"
LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
MODELS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"
KMOUNTS=(
  -v "$KERNELS_DIR/sparse_mla_kernels.py:$MLA/sparse_mla_kernels.py:ro"
  -v "$KERNELS_DIR/sparse_mla_env.py:$MLA/sparse_mla_env.py:ro"
  -v "$KERNELS_DIR/sm12x_sparse_mla_attn.py:$MLA/sm12x_sparse_mla_attn.py:ro"
  -v "$KERNELS_DIR/patch_flashmla_ops.py:$MLA/patch_flashmla_ops.py:ro"
  -v "$KERNELS_DIR/flashmla_sparse.py:$MLA/flashmla_sparse.py:ro"
  -v "$KERNELS_DIR/sm12x_deep_gemm_fallbacks.py:$OPS/sm12x_deep_gemm_fallbacks.py:ro"
  -v "$KERNELS_DIR/sm12x_mqa.py:$OPS/sm12x_mqa.py:ro"
  -v "$KERNELS_DIR/b12x_sparse_helpers.py:$OPS/b12x_sparse_helpers.py:ro"
  # upstream vLLM #46862: fused indexer Q rope+fp8-quant (fused_indexer_q_rope_quant)
  -v "$KERNELS_DIR/sparse_attn_indexer.py:$LAYERS/sparse_attn_indexer.py:ro"
  -v "$KERNELS_DIR/deepseek_v2.py:$MODELS/deepseek_v2.py:ro"
)

# docker run base — IB passthrough is REQUIRED (without --device=/dev/infiniband
# + IPC_LOCK + memlock, NCCL silently drops to TCP: ~12 vs 30+ tok/s).
BASE=(
  --restart unless-stopped  # issue #6 self-heal: engine death -> all nodes bounce + re-rendezvous
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  -v "$COMPILE_CACHE_DIR:/compile-cache"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)

# The serving command (mirrors the recipe's `command:`, with {port} resolved).
SERVE=(
  vllm serve /cache/huggingface/hub/glm52-int4-int8mix
  --served-model-name glm-5.2 --host 0.0.0.0 --port "$PORT"
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --enable-prefix-caching
  --async-scheduling
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}'
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  --max-model-len 316000 --max-num-seqs 6 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.91 --kv-cache-memory-bytes 10950000000
  --kv-cache-dtype nvfp4_ds_mla
  --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[6,12,18,24,30,36],"pass_config":{"fuse_gemm_comms":true}}'
)

# Build the full `docker run` for a given rank, as a single shell-quoted string.
docker_run_cmd() {
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}" "${KMOUNTS[@]}"
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
             "$IMAGE" "${SERVE[@]}"
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  # printf %q on each token yields a paste-safe, correctly-quoted command line.
  local out="" t
  for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done
  printf '%s' "${out# }"
}

say "GLM-5.2 launch: ${NNODES} nodes, head=$HEAD:$PORT, image=$IMAGE"
[ "$DRYRUN" = 1 ] && echo "   (dry-run — nothing will be executed)"

# Workers first (rank 1..N-1, headless), then the head (rank 0).
for ((rank=1; rank<NNODES; rank++)); do
  w="${NODES[$rank]}"
  run="$(docker_run_cmd "$rank" 1)"
  shell="docker rm -f $NAME 2>/dev/null; $run"
  if [ "$DRYRUN" = 1 ]; then
    printf '\n# worker %s (rank %d, headless)\nssh %s@%s %q\n' "$w" "$rank" "$SSH_USER" "$w" "$shell"
  else
    printf '   worker %s rank=%d (headless)\n' "$w" "$rank"
    ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$(node_user "$w")@$w" "$shell" \
      || die "worker launch failed on $w"
  fi
done

run="$(docker_run_cmd 0 0)"
shell="docker rm -f $NAME 2>/dev/null; $run"
if [ "$DRYRUN" = 1 ]; then
  printf '\n# head %s (rank 0)\n%s\n' "$HEAD" "$shell"
  exit 0
fi
printf '   head %s rank=0\n' "$HEAD"
bash -c "$shell" || die "head launch failed"

say "launched"
echo "   poll:  curl -s http://localhost:$PORT/v1/models"
echo "   logs:  docker logs -f $NAME   (on the head node)"
echo "   stop:  ./launch.sh --stop"
echo "   Ready in ~7 min load (warm) + ~5 min KV/graphs; serves as 'glm-5.2' (NVFP4, 316K)."
