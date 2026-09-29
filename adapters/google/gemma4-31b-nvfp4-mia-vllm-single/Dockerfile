FROM docker.io/vllm/vllm-openai@sha256:2cc49b81319f7a66a33dd8bd63a7bfddae079122b33ce51989b6828a1f038c37

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Gemma-4-31B-IT-NVFP4-Recipe" \
      org.opencontainers.image.revision="ac429cca70aedfa4afb446cefc5f5f1bfa1e8802" \
      org.opencontainers.image.licenses="Apache-2.0"

# Upstream bind-mounts its chat_template.jinja at run time; bake the same
# byte-identical file in instead.
COPY chat_template.jinja /opt/vonk/chat_template.jinja
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py
COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm

RUN python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
