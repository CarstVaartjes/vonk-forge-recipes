# syntax=docker/dockerfile:1.7
FROM ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark" \
      org.opencontainers.image.revision="fdcd538fbf95fb15b2d6850db9613d22b2c889b8" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.description="Offline Vonk adapter for MiaAI-Lab's 384K compact-NVFP4 one-DGX-Spark profile"

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
           --shell /usr/sbin/nologin vonk; \
       fi \
    && install --directory --owner 10001 --group 10001 \
         /opt/vonk/bin /opt/vonk/tiktoken-cache /state /outputs \
    && if test -d /root/.cache/tiktoken_cache; then \
         cp -a /root/.cache/tiktoken_cache/. /opt/vonk/tiktoken-cache/; \
       fi \
    && chmod -R a+rX /opt/vonk/tiktoken-cache

COPY overlays/coalesce_rank_sliced_exl3.py /opt/recipe/scripts/coalesce_rank_sliced_exl3.py
COPY overlays/sparkinfer/moe/fused_moe/_impl.py /opt/sparkinfer/sparkinfer/moe/fused_moe/_impl.py
COPY overlays/sparkinfer/moe/_shared/kernels/tiny_decode.py /opt/sparkinfer/sparkinfer/moe/_shared/kernels/tiny_decode.py
COPY overlays/vllm/models/deepseek_v4/nvidia/dspark.py /opt/vllm/vllm/models/deepseek_v4/nvidia/dspark.py
COPY overlays/sparkinfer/attention/_shared/mla/prefill.py /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill.py
COPY overlays/sparkinfer/attention/_shared/mla/prefill_mg.py /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill_mg.py
COPY --chmod=0755 overlays/serve-ds4-flash.sh /opt/vllm/serve-ds4-flash.sh
COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm
COPY NOTICE /opt/vonk/NOTICE.mia-sparkinfer-single

ENV HOME=/state \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TIKTOKEN_CACHE_DIR=/opt/vonk/tiktoken-cache \
    TIKTOKEN_RS_CACHE_DIR=/opt/vonk/tiktoken-cache \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
