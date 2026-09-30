FROM nvcr.io/nvidia/tensorrt-llm/release@sha256:82c5900384ae3463ea8a47f24c1a11fa3224ba1bb4bfebe0c7fbca22e785cb7c

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TensorRT-LLM" \
      org.opencontainers.image.revision="61cef212a8c59e843521881f45eee262c8f0525d" \
      org.opencontainers.image.version="1.3.0rc12" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --chmod=0755 trtllm-serve /opt/vonk/bin/trtllm-serve
COPY llm-api-config.yml /opt/vonk/trtllm/llm-api-config.yml
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
