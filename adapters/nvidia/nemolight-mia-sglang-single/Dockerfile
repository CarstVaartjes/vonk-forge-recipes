FROM docker.io/lmsysorg/sglang@sha256:ee63304703432429d55cf2e4579b70c5a334ed4cc49194fb5290c077fcbbd4a0

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Nemotron3.5-Lightning-DGX-Spark-RTX-5090-6000-PRO" \
      org.opencontainers.image.revision="38c16a403ea57f26f1828fa572270245ae51f01d" \
      org.opencontainers.image.licenses="MIT"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/state XDG_CACHE_HOME=/outputs/cache TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
