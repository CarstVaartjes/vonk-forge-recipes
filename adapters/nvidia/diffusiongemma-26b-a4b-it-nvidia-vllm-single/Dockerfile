FROM docker.io/vllm/vllm-openai@sha256:e7d5835566d26faaefa2f7b0f9c365e8bda83aa340f2ebf2f9ba8e53ed5c7e55

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.version="1.0.0" \
      io.vonk.upstream-image="vllm/vllm-openai:gemma (DiffusionGemma custom container named by the playbook)"

COPY --chmod=0755 vllm /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py
RUN python3 /opt/vonk/verify-runtime.py

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
