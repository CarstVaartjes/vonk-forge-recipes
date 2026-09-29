#!/usr/bin/env bash
# Puzzle-75B-A9B **NVFP4** (53.5GB, modelopt MIXED_PRECISION) — SINGLE GB10 (.3), TP=1.
# Frees .4 entirely vs the BF16 TP=2 builds. fi614 image (jethac A4Q + aeon 0.24).
# Usage: puzzle-nvfp4-serve.sh   Knobs: A4Q(1) KVD(nvfp4|fp8) SPEC(mtp) SPEC_TOKENS(3)
#   SEQS(32) MAXLEN(131072) MOE_BACKEND(marlin) EAGER(1)
set -uo pipefail
A4Q="${A4Q:-1}"; KVD="${KVD:-nvfp4}"; SPEC="${SPEC:-mtp}"; SPEC_TOKENS="${SPEC_TOKENS:-3}"
SEQS="${SEQS:-32}"; MAXLEN="${MAXLEN:-131072}"; MOE_BACKEND="${MOE_BACKEND:-marlin}"; EAGER="${EAGER:-1}"
[ "$KVD" != "nvfp4" ] && A4Q=0
EAGERFLAG=""; [ "$EAGER" = "1" ] && EAGERFLAG="--enforce-eager"  # marlin+CUDA graphs livelock on sm_121
SNAP=$(ls -d "$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/snapshots"/*/ | head -1)
cd "$SNAP" && for f in *.py *.json *.jinja; do [ -L "$f" ] && cp -L "$f" "$f.real" && mv "$f.real" "$f"; done; cd - >/dev/null
CSNAP="/root/${SNAP#$HOME/}"
SPECARG=""; [ "$SPEC" = "mtp" ] && SPECARG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}'"
bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
docker rm -f puzzle_nvfp4 puzzle_a4q puzzle75b >/dev/null 2>&1 || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --memory 112g --memory-swap 112g --ulimit memlock=-1 --ulimit nofile=1048576 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  --name puzzle_nvfp4 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=6 \
  -e VLLM_NVFP4_A4Q="$A4Q" -e VLLM_ATTENTION_BACKEND=FLASHINFER -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e HF_HUB_OFFLINE=1 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  --entrypoint bash hy3-arm3-aeon-fi614:test \
  -lc "
    exec vllm serve $CSNAP \
      --served-model-name puzzle-75b puzzle-nvfp4 --host 0.0.0.0 --port 8000 \
      --trust-remote-code --tensor-parallel-size 1 \
      --enable-expert-parallel --mamba-backend flashinfer \
      --moe-backend $MOE_BACKEND --kv-cache-dtype $KVD \
      --max-model-len $MAXLEN --max-num-seqs $SEQS --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.85 $EAGERFLAG $SPECARG \
      --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --enable-auto-tool-choice \
      --enable-prefix-caching
  "
echo "launched puzzle_nvfp4 single-node A4Q=$A4Q kv=$KVD moe=$MOE_BACKEND eager=$EAGER spec=$SPEC/$SPEC_TOKENS rc=$?"
