FROM ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s" \
      org.opencontainers.image.revision="34e81562984bda993e0c9ed01ed6900c17e4857b" \
      org.opencontainers.image.licenses="Apache-2.0"

# Upstream's port/build-nvfp4.sh starts from vllm-node-tf5-glm52-b12x:probe-modded (the
# published probe-modded tag of this base), copies the NVFP4 kernels and patched b12x
# files into the image, runs apply_nvfp4_plumbing.py --image, and separately patches a copy
# of the "KNOWNGOOD" sparse-MLA kernel overlay that it bind-mounts at run time. The overlay
# is the ten-file set of tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s at
# 162d7a76 (quanttrio-kernels/, see NOTICE.upstream); the patcher's overlay mode applies to
# it unchanged. Do the same at build time: patch the overlay copy, install it at the paths
# upstream bind-mounts, then the image-side files and plumbing.
COPY quanttrio-kernels /opt/vonk/build/kernels
COPY port /opt/vonk/build/port
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY install-nccl.py verify-runtime.py /opt/vonk/build/

RUN set -eu; \
    SP=/usr/local/lib/python3.12/dist-packages; \
    MLA="$SP/vllm/v1/attention/backends/mla"; \
    OPS="$SP/vllm/v1/attention/ops/deepseek_v4_ops"; \
    K=/opt/vonk/build/kernels; P=/opt/vonk/build/port; \
    python3 "$P/apply_nvfp4_plumbing.py" --overlay "$K"; \
    cp "$P/b12x-nvfp4/b12x_sparse_helpers.py" "$K/b12x_sparse_helpers.py"; \
    python3 -m py_compile "$K"/*.py; \
    for f in sparse_mla_kernels sparse_mla_env sm12x_sparse_mla_attn patch_flashmla_ops flashmla_sparse; do cp "$K/$f.py" "$MLA/$f.py"; done; \
    for f in sm12x_deep_gemm_fallbacks sm12x_mqa b12x_sparse_helpers; do cp "$K/$f.py" "$OPS/$f.py"; done; \
    cp "$K/sparse_attn_indexer.py" "$SP/vllm/model_executor/layers/sparse_attn_indexer.py"; \
    cp "$K/deepseek_v2.py" "$SP/vllm/model_executor/models/deepseek_v2.py"; \
    cp "$P/nvfp4_glm_kernels.py" "$MLA/nvfp4_glm_kernels.py"; \
    mkdir -p "$SP/b12x/attention/mla"; \
    for f in "$P"/b12x-nvfp4/*.py; do b=$(basename "$f"); [ "$b" = b12x_sparse_helpers.py ] || cp "$f" "$SP/b12x/attention/mla/$b"; done; \
    python3 "$P/apply_nvfp4_plumbing.py" --image "$SP/vllm" | tee /tmp/nvfp4-plumbing.log; \
    grep -q "PLUMBING OK" /tmp/nvfp4-plumbing.log; \
    python3 /opt/vonk/build/install-nccl.py; \
    python3 /opt/vonk/build/verify-runtime.py; \
    rm -rf /opt/vonk/build /tmp/nvfp4-plumbing.log; \
    install --directory --owner 10001 --group 10001 /state /outputs; \
    test -x /opt/vonk/bin/vllm; test -r /opt/vonk/lib/libnccl.so.2

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
