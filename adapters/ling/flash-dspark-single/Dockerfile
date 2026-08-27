FROM docker.io/lmsysorg/sglang@sha256:fc960102f1d2581b7dfa44885f6afcdfb107e84782e318cf33bbb2f219d0bd16

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Ling-3.0-Flash-SGLang-DSpark-DGX-Spark" \
      org.opencontainers.image.revision="ca840cb8d032353e24648aeee06312b0938348f6" \
      org.opencontainers.image.licenses="MIT"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/state XDG_CACHE_HOME=/outputs/cache TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
