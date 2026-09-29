#!/usr/bin/env bash
# Puzzle-75B NVFP4 — jvr0x catch-up profile (single GB10).
# Matches https://github.com/jvr0x/dgx-spark-bench/tree/main/recipes/nemotron-labs-3-puzzle-75b-nvfp4-aeon7
#   cutlass MoE + CUDA graphs + async + BTOK 16k + fp8 KV + MTP k=3 + max-len 32k
# Usage: puzzle-jvr-catchup-serve.sh
# Knobs: IMAGE PORT SEQS MAXLEN BTOK EAGER(0) KVD(fp8) SPEC_TOKENS(3) GMU(0.85)
set -uo pipefail

IMAGE="${IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"
PORT="${PORT:-8243}"
SEQS="${SEQS:-64}"
MAXLEN="${MAXLEN:-32768}"
BTOK="${BTOK:-16384}"
EAGER="${EAGER:-0}"
KVD="${KVD:-fp8}"
SPEC_TOKENS="${SPEC_TOKENS:-3}"
GMU="${GMU:-0.85}"
PREFIX="${PREFIX:-1}"          # 0 = no prefix cache (bench unique prompts)
THINKING="${THINKING:-0}"      # 0 = default enable_thinking false
NAME="${NAME:-puzzle_jvr}"

SNAP=$(ls -d "$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/snapshots"/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
  echo "ERROR: Puzzle NVFP4 snapshot not found under HF cache" >&2
  exit 1
fi
# Materialize small-file symlinks (offline trust_remote_code trap)
cd "$SNAP" && for f in *.py *.json *.jinja; do
  [ -L "$f" ] && cp -L "$f" "$f.real" && mv "$f.real" "$f"
done; cd - >/dev/null
CSNAP="/root/${SNAP#$HOME/}"

EAGERFLAG=""
[ "$EAGER" = "1" ] && EAGERFLAG="--enforce-eager"
PREFIXFLAG=""
[ "$PREFIX" = "1" ] && PREFIXFLAG="--enable-prefix-caching"
# Thinking off by default so bench TTFT lands on content, not a long think stream.
THINKFLAG=""
if [ "$THINKING" = "0" ]; then
  # Escaped for the container bash -lc double-quoted block
  THINKFLAG='--default-chat-template-kwargs {\"enable_thinking\":false}'
fi

bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
docker rm -f "$NAME" puzzle_nvfp4 puzzle_a4q puzzle75b >/dev/null 2>&1 || true

docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --memory 112g --memory-swap 112g --ulimit memlock=-1 --ulimit nofile=1048576 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  --name "$NAME" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e CUTE_DSL_ARCH=sm_121a \
  -e MAX_JOBS=6 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  --entrypoint bash "$IMAGE" \
  -lc "
    exec vllm serve $CSNAP \
      --served-model-name puzzle-jvr nemotron-labs-3-puzzle-75b-nvfp4-aeon7 puzzle-75b \
      --host 0.0.0.0 --port $PORT \
      --trust-remote-code --tensor-parallel-size 1 \
      --enable-expert-parallel \
      --mamba-backend flashinfer \
      --moe-backend flashinfer_cutlass \
      --kv-cache-dtype $KVD \
      --max-model-len $MAXLEN \
      --max-num-seqs $SEQS \
      --max-num-batched-tokens $BTOK \
      --gpu-memory-utilization $GMU \
      $EAGERFLAG \
      --async-scheduling \
      --enable-chunked-prefill \
      $PREFIXFLAG \
      --generation-config vllm \
      --override-generation-config '{\"temperature\":0.0,\"top_p\":1.0}' \
      $THINKFLAG \
      --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}' \
      --tool-call-parser qwen3_coder \
      --reasoning-parser nemotron_v3 \
      --enable-auto-tool-choice
  "
echo "launched $NAME image=$IMAGE port=$PORT eager=$EAGER kvd=$KVD maxlen=$MAXLEN btok=$BTOK seqs=$SEQS prefix=$PREFIX thinking=$THINKING gmu=$GMU rc=$?"
echo "logs: docker logs -f $NAME"
