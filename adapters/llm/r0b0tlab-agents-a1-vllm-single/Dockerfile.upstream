FROM ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121:latest

LABEL org.opencontainers.image.title="Agents-A1 NVFP4 SM121 vLLM"
LABEL org.opencontainers.image.description="SM121-native vLLM runtime for InternScience/Agents-A1 NVFP4 ModelOpt checkpoints"
LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/agents-a1-nvfp4-sm121-vllm"
LABEL org.opencontainers.image.licenses="Apache-2.0"

USER root

# Strict publish gate: this derivative image removes packaged fallback FlashAttention
# extensions that are not needed for the Agents-A1 text-only SM121 NVFP4 path.
# Marlin artifacts are also removed if ever inherited from a base image.
RUN set -eux; \
    rm -f /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C*.so; \
    rm -f /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa3_C*.so; \
    find /usr/local/lib/python3.12/dist-packages -iname '*marlin*.so' -delete; \
    /usr/bin/python3 - <<'PY'
from pathlib import Path
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/__init__.py')
s = p.read_text()
old = '''if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
    )
'''
new = '''if not (FA2_AVAILABLE or FA3_AVAILABLE):
    # SM121 NVFP4 text-only images intentionally omit FA2/FA3 fallback
    # extension artifacts. FlashInfer is selected as the active attention
    # backend; leaving this importable lets model modules that reference
    # fa_utils load without activating FA2/FA3 kernels.
    pass
'''
p.write_text(s.replace(old, new))
PY

COPY scripts/start_agents_a1.sh /usr/local/bin/start_agents_a1.sh
COPY scripts/healthcheck.sh /usr/local/bin/healthcheck.sh
COPY scripts/audit_runtime.py /usr/local/bin/audit_runtime.py
RUN chmod +x /usr/local/bin/start_agents_a1.sh /usr/local/bin/healthcheck.sh /usr/local/bin/audit_runtime.py

ENV MODEL_ID="/models/Agents-A1-NVFP4"
ENV SERVED_MODEL_NAME="agents-a1-nvfp4"
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV MAX_MODEL_LEN="4096"
ENV MAX_NUM_SEQS="32"
ENV MAX_NUM_BATCHED_TOKENS="8192"
ENV GPU_MEMORY_UTILIZATION="0.70"
ENV KV_CACHE_DTYPE="fp8"
ENV QUANTIZATION="modelopt"
ENV ATTENTION_BACKEND="flashinfer"
ENV MOE_BACKEND="flashinfer_cutlass"
ENV TOOL_CALL_PARSER="qwen3_coder"
ENV REASONING_PARSER="qwen3"
ENV SPECULATIVE_CONFIG=""
ENV EXTRA_ARGS=""

# SM121 / GB10 native NVFP4 runtime defaults.
ENV FLASHINFER_DISABLE_VERSION_CHECK="1"
ENV VLLM_NVFP4_GEMM_BACKEND="flashinfer-cutlass"
ENV VLLM_FP8_MOE_BACKEND="flashinfer_cutlass"
ENV CUTE_DSL_ARCH="sm_121a"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV VLLM_LOGGING_LEVEL="INFO"

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD /usr/local/bin/healthcheck.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/start_agents_a1.sh"]
