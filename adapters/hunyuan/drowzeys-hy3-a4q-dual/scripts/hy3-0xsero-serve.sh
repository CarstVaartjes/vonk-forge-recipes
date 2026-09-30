#!/usr/bin/env bash
# Hy3 replication of joeynyc/Hy3-295B-NVFP4-2x-DGX-Spark on our fabric.
# 0xSero/Hy3-299B-NVFP4 (compressed-tensors nvfp4-pack-quantized) on nvcr.io/nvidia/vllm:26.06-py3.
# CUDA graphs ON (no eager), TurboQuant k8v4 KV, 256K, TP=2 (.1 rank0 API :8000, .2 rank1).
# Mods baked at launch: fp32 expert_bias (vllm#47777), MTP eh_proj quant fix (vllm#47792, for SPEC=mtp legs).
# Usage: hy3-0xsero-serve.sh <rank 0|1>  Knobs: SPEC(none|mtp) SPEC_TOKENS(2) DRAFT_TP() SEQS(4) MAXLEN(262144) KVD(turboquant_k8v4) EAGER(0) BACKEND(mp|ray)
set -uo pipefail
RANK="${1:?}"; MASTER=10.100.10.1; PORT=29550; IF=enp1s0f1np1; HCA=rocep1s0f1
SPEC="${SPEC:-none}"; SPEC_TOKENS="${SPEC_TOKENS:-2}"; DRAFT_TP="${DRAFT_TP:-}"
SEQS="${SEQS:-4}"; MAXLEN="${MAXLEN:-262144}"; KVD="${KVD:-turboquant_k8v4}"; EAGER="${EAGER:-0}"
SELF=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELF=${SELF:-$MASTER}
HEADLESS=""; [ "$RANK" != "0" ] && HEADLESS="--headless"
EAGERFLAG=""; [ "$EAGER" = "1" ] && EAGERFLAG="--enforce-eager"
MODEL="${MODEL:-0xsero}"
case "$MODEL" in
  libertai) REPO="models--LibertAIDAI--Hy3-NVFP4" ;;
  *)        REPO="models--0xSero--Hy3-299B-NVFP4" ;;
esac
SNAP=$(ls -d "$HOME/.cache/huggingface/hub/$REPO/snapshots"/*/ | head -1)
# 0xSero tokenizer uses OLD hy_v3 token names -> stock nvcr parsers match; LibertAI needs :opensource overlays
PARSER_MOUNTS=""
[ "$MODEL" = "libertai" ] && PARSER_MOUNTS='-v '"$HOME"'/hy3-overlays/hy_v3_tool_parser.py:/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/hy_v3_tool_parser.py:ro -v '"$HOME"'/hy3-overlays/hy_v3_reasoning_parser.py:/usr/local/lib/python3.12/dist-packages/vllm/reasoning/hy_v3_reasoning_parser.py:ro' 
CSNAP="/root/${SNAP#$HOME/}"
SPECJSON="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS"
[ -n "$DRAFT_TP" ] && SPECJSON="$SPECJSON,\"draft_tensor_parallel_size\":$DRAFT_TP"
SPECJSON="$SPECJSON}"
SPECARG=""; [ "$SPEC" = "mtp" ] && SPECARG="--speculative-config '$SPECJSON'"
bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true  # no-reboot memory hygiene
docker rm -f hy3_0xsero hy3_a4q hy3_nightly >/dev/null 2>&1 || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --memory 112g --memory-swap 112g --ulimit memlock=-1 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  $PARSER_MOUNTS \
  -v "$HOME/hy3-overlays-nvcr/compressed_tensors_moe_w4a4_nvfp4.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py:ro" \
  --name hy3_0xsero -e MODELTAG=$MODEL \
  -e VLLM_HOST_IP=$SELF -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash nvcr.io/nvidia/vllm:26.06-py3 \
  -lc '
    SITE=$(python3 -c "import vllm,os; print(os.path.dirname(vllm.__file__))")
    # joeynyc mod 1 (vllm#47777): expert_bias must stay fp32
    sed -i "s/self.expert_bias = nn.Parameter(torch.empty(config.num_experts))/self.expert_bias = nn.Parameter(torch.empty(config.num_experts, dtype=torch.float32))/" $SITE/model_executor/models/hy_v3.py 2>/dev/null
    # joeynyc mod 2 (vllm#47792): MTP eh_proj must be quant-aware (only matters for SPEC=mtp)
    python3 - <<PYEOF
p = "$SITE/model_executor/models/hy_v3_mtp.py"
src = open(p).read()
if "ReplicatedLinear" not in src:
    src = src.replace("from vllm.model_executor.layers.layernorm import RMSNorm",
        "from vllm.model_executor.layers.layernorm import RMSNorm\nfrom vllm.model_executor.layers.linear import ReplicatedLinear", 1)
    for old in ("self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)",
                "self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)"):
        if old in src:
            src = src.replace(old, "self.eh_proj = ReplicatedLinear(config.hidden_size * 2, config.hidden_size, bias=False, quant_config=quant_config, prefix=f\"{prefix}.eh_proj\")", 1)
            break
    src = src.replace("""        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )""", """        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )""", 1)
    open(p, "w").write(src)
    print("eh_proj fix applied (ctor + call site)")
PYEOF
    exec vllm serve '"$CSNAP"' \
      --served-model-name hy3-nvfp4 hy3-0xsero --host 0.0.0.0 --port 8000 \
      --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --kv-cache-dtype '"$KVD"' --block-size 64 \
      --max-model-len '"$MAXLEN"' --max-num-seqs '"$SEQS"' --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.85 '"$EAGERFLAG"' '"$SPECARG"' \
      --enable-prefix-caching --enable-auto-tool-choice \
      --tool-call-parser hy_v3 --reasoning-parser hy_v3 \
      --distributed-executor-backend mp \
      --nnodes 2 --node-rank '"$RANK"' --master-addr '"$MASTER"' --master-port '"$PORT"' '"$HEADLESS"'
  '
echo "launched hy3_0xsero model=$MODEL rank=$RANK kv=$KVD eager=$EAGER spec=$SPEC/$SPEC_TOKENS draft_tp=${DRAFT_TP:-full} rc=$?"
