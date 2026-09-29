FROM ghcr.io/drowzeys/vllm-dsv41-overlay5-e47aa@sha256:ad5cc20ca16f30ac6d4364cbc2ab00785b1575d31b90488d026d4245eead19a3

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-8M-KV-Four-DGX-Sparks-with-abliterated-option" \
      org.opencontainers.image.revision="4b69803bf6eca380a775e40b322ac2542dae3a59" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# Base: upstream's prebuilt engine image (tag serving-node1: vLLM dsv41-feat
# e47aa780b with DSpark and the TR3/B12X plugin, GB10 sm_121a). Upstream
# bind-mounts the seven patch files (patch-upstream-boot10/mounts.txt) and the
# hybrid-port plugin tree (as /brandon) at container start; copy them in here
# so the root filesystem can stay read-only.
COPY serve/patch-upstream-boot10/sparse_swa.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py
COPY serve/patch-upstream-boot10/attention.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/attention.py
COPY serve/patch-upstream-boot10/flashinfer_sparse.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py
COPY serve/patch-upstream-boot10/engram.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py
COPY serve/patch-upstream-boot10/weight_utils.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py
COPY serve/patch-upstream-boot10/model_state.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py
COPY serve/patch-upstream-boot10/sparse_attn_indexer.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
COPY serve/hybrid-port /opt/brandon
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /opt/vonk/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton \
    && install --directory /opt/engram-empty

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    PYTHONPATH=/opt/brandon:/opt/brandon/plugin-root \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    HF_HOME=/outputs/cache/huggingface \
    TILELANG_CACHE_DIR=/outputs/cache/tilelang

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
