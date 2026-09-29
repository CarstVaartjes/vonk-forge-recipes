#!/usr/bin/env bash
# Hy3 (Tencent 295B-A21B MoE) LibertAIDAI/Hy3-NVFP4 (modelopt weight-only NVFP4 experts)
# Fresh vLLM nightly (0.24 line) with hy_v3 parsers. TP=2: .1 = rank0 (API :8000), .2 = rank1 headless.
# Recipe basis: LibertAIDAI deploy/ (verified 2xGB10: marlin MoE + eager + fp8 KV + 256K + prefix cache)
# + our fabric conventions (RoCE ifname, GID 3, mp --nnodes 2).
# Usage: hy3-nightly-serve.sh <node_rank 0|1>
#   Knobs: IMAGE MAXLEN(262144) UTIL(0.90) MAXSEQ(32) BTOK(16384) KVD(fp8) SPEC(none|mtp) SPEC_TOKENS(2) PARSERS(1) REASONING(1)
set -uo pipefail
RANK="${1:?need rank 0 or 1}"
MASTER=10.100.10.1; PORT=29530; IF=enp1s0f1np1; HCA=rocep1s0f1
IMAGE="${IMAGE:-hy3-nightly:fresh}"
MAXLEN="${MAXLEN:-163840}"; UTIL="${UTIL:-0.85}"; MAXSEQ="${MAXSEQ:-32}"; BTOK="${BTOK:-16384}"  # UTIL 0.85 = universal cluster rule (recipe's 0.90 overridden)
KVD="${KVD:-fp8}"; SPEC="${SPEC:-none}"; SPEC_TOKENS="${SPEC_TOKENS:-2}"; PARSERS="${PARSERS:-1}"; REASONING="${REASONING:-1}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"   # only NVFP4-MoE backend alive on sm_121a (deploy README dead-ends table)
SELF=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELF=${SELF:-$MASTER}
HEADLESS=""; [ "$RANK" != "0" ] && HEADLESS="--headless"
SPECARG=""; [ "$SPEC" = "mtp" ] && SPECARG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}'"
PARSERARG=""
[ "$PARSERS" = "1" ] && PARSERARG="--tool-call-parser hy_v3 --enable-auto-tool-choice"
[ "$PARSERS" = "1" ] && [ "$REASONING" = "1" ] && PARSERARG="$PARSERARG --reasoning-parser hy_v3"
bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
docker rm -f hy3_nightly >/dev/null 2>&1 || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --memory 112g --memory-swap 112g \
  --ulimit memlock=-1 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  --name hy3_nightly \
  -e VLLM_HOST_IP=$SELF -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=6 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e HF_HUB_OFFLINE=1 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  --entrypoint bash "$IMAGE" \
  -lc "
    exec vllm serve LibertAIDAI/Hy3-NVFP4 \
      --served-model-name hy3-nvfp4 hy3-arm3 --host 0.0.0.0 --port 8000 \
      --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --moe-backend $MOE_BACKEND --enforce-eager \
      --kv-cache-dtype $KVD --max-model-len $MAXLEN --max-num-seqs $MAXSEQ \
      --max-num-batched-tokens $BTOK --gpu-memory-utilization $UTIL \
      --enable-prefix-caching $SPECARG $PARSERARG \
      --distributed-executor-backend mp \
      --nnodes 2 --node-rank $RANK --master-addr $MASTER --master-port $PORT $HEADLESS
  "
echo "launched hy3_nightly rank=$RANK image=$IMAGE moe=$MOE_BACKEND seqs=$MAXSEQ rc=$?"
