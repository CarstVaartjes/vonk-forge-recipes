# syntax=docker/dockerfile:1.7
FROM ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4

# Upstream d370c0f0 installs XGrammar 0.2.3 (normalize_tool_choice, required by the
# pinned vLLM tool parser) in its Dockerfile, but its published image digest is
# unchanged, so the pinned wheel is installed here, hash-checked.
RUN printf 'xgrammar==0.2.3 --hash=sha256:11255f184971489fc72b948b096e2917f482ba2dca975177f5411562cedb9c6d\n' > /tmp/xgrammar.txt \
    && env -u PIP_CONSTRAINT /opt/runtime-venv/bin/python -m pip install \
         --no-deps --only-binary :all: --require-hashes -r /tmp/xgrammar.txt \
    && /opt/runtime-venv/bin/python -c \
         'import importlib.metadata; from xgrammar import normalize_tool_choice; assert importlib.metadata.version("xgrammar") == "0.2.3"' \
    && rm -f /tmp/xgrammar.txt

LABEL org.opencontainers.image.source="https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer" \
      org.opencontainers.image.revision="d370c0f0806a9ac994dc5f354715576cc23840b9" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Offline Vonk adapter for the validated one-DGX-Spark SparkInfer profile"

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
COPY NOTICE /opt/vonk/NOTICE.sparkinfer-single

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
