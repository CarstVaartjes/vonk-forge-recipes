FROM ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s" \
      org.opencontainers.image.revision="162d7a766e4bc9de1809ff724339538380c43b0c" \
      org.opencontainers.image.licenses="Apache-2.0"

# The base is the published `probe-modded` tag: the vLLM ab666069 (0.23.1rc1.dev190)
# build with the two glm52 sparse mods applied, the image upstream calls
# vllm-node-tf5-glm52-b12x:probe-modded. Upstream bind-mounts its matched set of ten
# kernel files over the vLLM tree at run time (GLM52_BIND_HOST_TRITON=1); copy the same
# files to the same paths at build time, apply the indexer MTP-overhang patch upstream
# bakes into its image, and bake the optional decode-aware scheduler mod (inactive unless
# its flags are passed).
COPY kernels/sparse_mla_kernels.py kernels/sparse_mla_env.py kernels/sm12x_sparse_mla_attn.py kernels/patch_flashmla_ops.py kernels/flashmla_sparse.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/
COPY kernels/sm12x_deep_gemm_fallbacks.py kernels/sm12x_mqa.py kernels/b12x_sparse_helpers.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops/
COPY kernels/sparse_attn_indexer.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
COPY kernels/deepseek_v2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py
COPY patches/fix-indexer-mtp-overhang.py /opt/vonk/build/fix-indexer-mtp-overhang.py
COPY mods/decode-aware-scheduler /opt/vonk/build/decode-aware-scheduler
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY install-nccl.py verify-runtime.py /opt/vonk/build/

RUN python3 /opt/vonk/build/fix-indexer-mtp-overhang.py \
    && bash /opt/vonk/build/decode-aware-scheduler/run.sh \
    && python3 /opt/vonk/build/install-nccl.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && install --directory --owner 10001 --group 10001 /state /outputs \
    && test -x /opt/vonk/bin/vllm \
    && test -r /opt/vonk/lib/libnccl.so.2

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
