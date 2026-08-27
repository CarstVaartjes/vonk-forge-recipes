FROM docker.io/lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark" \
      org.opencontainers.image.revision="751e29eb6a3057ccfd8f992f87dfc260787e05a1" \
      org.opencontainers.image.licenses="MIT"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/state XDG_CACHE_HOME=/outputs/cache TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
