FROM docker.io/lmsysorg/sglang@sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark" \
      org.opencontainers.image.revision="9fb18edf8cfb3364e8aa89258e6d5ab1fe1fd11a" \
      org.opencontainers.image.licenses="MIT"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/state XDG_CACHE_HOME=/outputs/cache TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
