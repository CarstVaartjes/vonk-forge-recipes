FROM ghcr.io/drowzeys/vllm-dsv41-overlay5-e47aa@sha256:ad5cc20ca16f30ac6d4364cbc2ab00785b1575d31b90488d026d4245eead19a3

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks" \
      org.opencontainers.image.revision="74ac01362e96c1dc72df96737354ce408dceb272" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# Base: drowzeys' published build of the same stack (vLLM dsv41-feat at e47aa780b, the
# tree every patch below targets, built for GB10 sm_121a). It is a third party's build,
# NOT the creator's overlay5 image, and a creator-built replacement (docs/recipe-authoring.md,
# "Missing creator image") is still owed; this recipe is blocked on that until then.
# The creator bind-mounts the patch files at container start (patch/mounts.txt);
# they are copied in here so the root filesystem can stay read-only. Six files
# are byte-identical to the creator's patch/ set; engram.py is the creator's
# speed-run Engram fast-staging file (dsv41-exl3-sr1, md5 e84c7305) that the
# 1017 boots 1017c and later mount.
COPY patch/sparse_swa.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py
COPY patch/attention.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/attention.py
COPY patch/flashinfer_sparse.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py
COPY patch/engram.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py
COPY patch/weight_utils.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py
COPY patch/model_state.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py
COPY patch/sparse_attn_indexer.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.tonyd2wild-dsv41
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /usr/local/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    HF_HOME=/outputs/cache/huggingface \
    TILELANG_CACHE_DIR=/outputs/cache/tilelang

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
