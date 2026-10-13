FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark" \
      org.opencontainers.image.revision="d66a95e1e20af459075beb7fb88943942b1e637f" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Upstream run.sh (default path: v0.30.0 image, OVERLAYS=auto, GDN_LAZY=1, FP8_DENSE=per_block,
# DRAFT_MOE_CONFIG=1) bind-mounts these R11 overlays read-only over the stock files; the recipe
# bakes the same byte-identical files at the same paths. gdn_lazy_attn.py replaces gdn_attn.py.
COPY docker/v030/flashinfer_cutlass_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py
COPY docker/v030/gdn_lazy_attn.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py
COPY docker/v030/gdn_lazy_linear_attn.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
COPY docker/v030/qsa_indexer.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
COPY docker/v030/serving.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
COPY docker/v030/mtp.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py
COPY docker/v030/modelopt.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
# Upstream mounts the drafter's GB10 Triton MoE config folder as VLLM_TUNED_CONFIG_FOLDER.
COPY docker/v030/moe_configs/ /opt/qwen38-moe-configs/
# Upstream mounts the template on the head; baked in, either rank can name it.
COPY chat_template_alias.jinja /opt/vonk/chat_template_alias.jinja
COPY --chmod=0755 qwen38-vllm-wrapper.py /opt/vonk/bin/vllm
COPY LICENSE /opt/vonk/LICENSE.upstream

RUN set -e; \
    sp=/usr/local/lib/python3.12/dist-packages/vllm; \
    grep -q 'gdl_decode' $sp/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py; \
    grep -q 'get_top_tokens' $sp/models/qwen4_exp/nvidia/mtp.py; \
    for f in model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py v1/attention/backends/gdn_attn.py model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py models/qwen4_exp/nvidia/ops/qsa_indexer.py entrypoints/openai/chat_completion/serving.py models/qwen4_exp/nvidia/mtp.py model_executor/layers/quantization/modelopt.py; do \
      grep -q '^# R11-OVERLAY$' "$sp/$f"; \
      chmod 0644 "$sp/$f"; \
      find "$(dirname "$sp/$f")" -name '__pycache__' -prune -exec rm -rf {} +; \
    done; \
    chmod -R a+rX /opt/qwen38-moe-configs /opt/vonk/chat_template_alias.jinja; \
    install --directory --mode=0755 /outputs; \
    if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi; \
    if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi; \
    chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
