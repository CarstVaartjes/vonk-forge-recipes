# syntax=docker/dockerfile:1.7
FROM ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4

LABEL org.opencontainers.image.source="https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer" \
      org.opencontainers.image.revision="590d2172394dd83c1f36ff29f0dc9ec6032ea9e2" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Offline Vonk target-only canary adapter for DeepSeek V4 Flash on one DGX Spark"

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

COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm
COPY NOTICE /opt/vonk/NOTICE.sparkinfer-target-only-single

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
