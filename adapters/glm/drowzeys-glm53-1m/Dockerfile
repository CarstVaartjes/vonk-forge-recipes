FROM ghcr.io/drowzeys/keys-vllm-glm53-flash-nvfp4-ablit@sha256:f722ec19d8260833e948d5bf46949d9ac574841860060caa24213cf550d1a41b

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated" \
      org.opencontainers.image.revision="c36b5958412158a69629e7fbed321312e6d0761d" \
      org.opencontainers.image.licenses="Apache-2.0"

# Upstream now publishes the exact validated GLM 5.3 CUDA stack with b12x 1.2.6
# and Cutlass DSL 4.7.0. Verify it instead of rebuilding that network layer on
# every Spark; the local source bundle below still binds every reviewed overlay.
RUN python3 -c "import b12x, cutlass; assert b12x and cutlass"

COPY overlays/official/attention.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/attention.py
COPY overlays/official/flashinfer_mla_sparse_sm120.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
COPY overlays/official/flashinfer_mla_sparse.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
COPY overlays/official/cuda.py /usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py
COPY overlays/official/sparse_attn_indexer_kpool.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py
COPY overlays/official/kpool_compress.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/ops/kpool_compress.py

COPY overlays/nvfp4-b12x/config/cache.py /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py
COPY overlays/nvfp4-b12x/config/vllm.py /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py
COPY overlays/nvfp4-b12x/envs.py /usr/local/lib/python3.12/dist-packages/vllm/envs.py
COPY overlays/nvfp4-b12x/model_executor/layers/attention/mla_attention.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py
COPY overlays/nvfp4-b12x/model_executor/layers/mla_cache_format.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla_cache_format.py
COPY overlays/nvfp4-b12x/utils/torch_utils.py /usr/local/lib/python3.12/dist-packages/vllm/utils/torch_utils.py
COPY overlays/nvfp4-b12x/v1/attention/backends/mla/b12x_mla_sparse.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
COPY overlays/nvfp4-b12x/v1/attention/backends/registry.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/registry.py
COPY overlays/nvfp4-b12x/v1/kv_cache_interface.py /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    KV_FP8_ROPE=1 \
    GLM53_PAD_ROPE=auto \
    VLLM_PLUGINS="" \
    VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    DG_JIT_CACHE_DIR=/outputs/cache/deep_gemm_nvdev

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
