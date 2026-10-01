FROM docker.io/vllm/vllm-openai@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b

LABEL org.opencontainers.image.source="https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark" \
      org.opencontainers.image.revision="fe8c291de4efba34d5dcedbc8e19ebcf66bc1bc2" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Upstream run.sh bind-mounts these two vLLM overlays read-only into the stock
# nightly image; the recipe bakes the same byte-identical files at the same paths.
COPY docker/ple_layer.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py
COPY docker/modelopt.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
COPY --chmod=0755 qwen38-vllm-wrapper.py /opt/vonk/bin/vllm
COPY LICENSE /opt/vonk/LICENSE.upstream

# Same checks as upstream run.sh ensure_ple_overlay / ensure_mtp_overlay.
RUN grep -q 'ModelOptMixedPrecisionConfig' /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py \
    && grep -q 'Qwen4ExpPLEFp8EmbeddingMethod' /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py \
    && grep -q 'FP8_BLOCK_SCALES' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py \
    && chmod 0644 /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py \
    && find /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization -name '__pycache__' -prune -exec rm -rf {} + \
    && install --directory --mode=0755 /outputs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
