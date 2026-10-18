FROM eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/local-ai-inference-recipes" \
      org.opencontainers.image.revision="ca3326c375f77ab60301898a83f3094ee72b0884"
COPY LICENSE /opt/vonk/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper /opt/vonk/bin/vllm
RUN python -c "import vllm" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
