#!/usr/bin/env bash
# Hy3 LibertAIDAI/Hy3-NVFP4 on aeon fi614 image with jethac A4Q + nvfp4 KV + MTP.
# Per drowzeys/Implementation-of-Jetha-Chan-A4Q...: A4Q = VLLM_NVFP4_A4Q=1 on aeon/jethac
# lineage, nvfp4 KV prerequisite; marker log "A4Q: nvf4 block-scaled QK MMA enabled".
# Usage: hy3-a4q-serve.sh <rank 0|1>  Knobs: MAXLEN(262144) UTIL(0.85) MAXSEQ(16) BTOK(2048) SPEC_TOKENS(2) A4Q(1)
set -uo pipefail
RANK="${1:?}"; MASTER=10.100.10.1; PORT=29540; IF=enp1s0f1np1; HCA=rocep1s0f1
MAXLEN="${MAXLEN:-262144}"; UTIL="${UTIL:-0.85}"; MAXSEQ="${MAXSEQ:-16}"; BTOK="${BTOK:-2048}"
A4Q="${A4Q:-1}"; KVD="${KVD:-nvfp4}"; SPEC_TOKENS="${SPEC_TOKENS:-2}"
[ "$KVD" != "nvfp4" ] && A4Q=0  # A4Q exists only on the nvfp4-KV FA2 path (arm3 probeA2 lesson)
SELF=$(ip -4 addr show $IF 2>/dev/null|awk "/inet /{print \$2}"|cut -d/ -f1); SELF=${SELF:-$MASTER}
HEADLESS=""; [ "$RANK" != "0" ] && HEADLESS="--headless"
SNAP=$(ls -d "$HOME/.cache/huggingface/hub/models--LibertAIDAI--Hy3-NVFP4/snapshots"/*/ | head -1)
CSNAP="/root/${SNAP#$HOME/}"
# MTP k=2 default since 2026-07-08 (hy_v3_mtp overlay fix); SPEC=none to disable
SPEC="${SPEC:-mtp}"; SPECARG=""; [ "$SPEC" = "mtp" ] && SPECARG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}'"
bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
docker rm -f hy3_a4q hy3_nightly >/dev/null 2>&1 || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --memory 112g --memory-swap 112g --ulimit memlock=-1 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/hy3-overlays/hy_v3_tool_parser.py:/usr/local/lib/python3.12/site-packages/vllm/tool_parsers/hy_v3_tool_parser.py:ro" \
  -v "$HOME/hy3-overlays/hy_v3_reasoning_parser.py:/usr/local/lib/python3.12/site-packages/vllm/reasoning/hy_v3_reasoning_parser.py:ro" \
  -v "$HOME/hy3-overlays/hy_v3_mtp.py:/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/hy_v3_mtp.py:ro" \
  --name hy3_a4q \
  -e VLLM_HOST_IP=$SELF -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=6 \
  -e VLLM_NVFP4_A4Q="$A4Q" -e VLLM_ATTENTION_BACKEND=FLASHINFER -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e HF_HUB_OFFLINE=1 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  --entrypoint bash hy3-arm3-aeon-fi614:test \
  -lc "
    ln -sf /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null
    FAR=/usr/local/lib/python3.12/site-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py
    sed -i s/^except ImportError:/except Exception:/ \$FAR 2>/dev/null
    exec vllm serve $CSNAP --served-model-name hy3-nvfp4 hy3-a4q --host 0.0.0.0 --port 8000 \
      --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --moe-backend marlin --kv-cache-dtype $KVD \
      --max-model-len $MAXLEN --gpu-memory-utilization $UTIL --max-num-seqs $MAXSEQ \
      --max-num-batched-tokens $BTOK --enforce-eager $SPECARG \
      --tool-call-parser hy_v3 --enable-auto-tool-choice --reasoning-parser hy_v3 \
      --distributed-executor-backend mp --nnodes 2 --node-rank $RANK \
      --master-addr $MASTER --master-port $PORT $HEADLESS
  "
echo "launched hy3_a4q rank=$RANK A4Q=$A4Q kv=$KVD len=$MAXLEN mtp=$SPEC_TOKENS rc=$?"
