FROM eugr/spark-vllm@sha256:4428520bf19839a81eaa9f0647e421713e7246bd9ca28669dd742cb09955ad86
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/local-ai-inference-recipes" \
      org.opencontainers.image.revision="228fed3e236036887dcbb1a2f09d4db3c71620f9"
COPY LICENSE /opt/vonk/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper /opt/vonk/bin/vllm
RUN python3 -c "import vllm" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
