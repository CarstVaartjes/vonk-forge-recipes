# syntax=docker/dockerfile:1.7
# 0xSero single-Spark GLM-5.3-Flash EXL3 (codebook mul1) on SGLang: the public arm64
# exl3-plain runtime image that upstream's docker/run-spark.sh names
# (2p05-sglang-mul1-r1, the image that served every receipted result), pinned by
# manifest digest. It is the lmsysorg/sglang glm-5.3-flash base with the exl3-plain
# overlay. Our layer adds the census-then-serve entrypoint and the non-root user.
FROM ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed

LABEL org.opencontainers.image.source="https://github.com/0xSero/glm-5.3-flash-spark-mosaic" \
      org.opencontainers.image.revision="248dbaa00fffab296f255c16546c21621a4b2d89" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="Offline Vonk adapter for the one-DGX-Spark GLM-5.3-Flash EXL3 SGLang profiles" \
      ai.vonkforge.runtime-interface="v1"

COPY docker /opt/vonk/upstream-docker
COPY LICENSE /opt/vonk/LICENSE.glm53-exl3-plain-single
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
           --shell /usr/sbin/nologin vonk; \
       fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && chmod -R a+rX /opt/vonk /opt/exl3-plain-overlay /opt/glm53 \
    && python3 -c 'import exl3_plain_sglang_overlay.census, sglang' \
    && test -r /opt/glm53/chat-template-mm.jinja \
    && test -x /opt/vonk/bin/sglang-serve

ENV HOME=/state \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
