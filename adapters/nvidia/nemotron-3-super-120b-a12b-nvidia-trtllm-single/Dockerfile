FROM nvcr.io/nvidia/tensorrt-llm/release@sha256:3f293455e9450298f548c5c3268186f80703dfc490926f398aec9a746422e840

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TensorRT-LLM" \
      org.opencontainers.image.revision="33533348196f236dc291ad573710184858501f79" \
      org.opencontainers.image.version="1.3.0rc9" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --chmod=0755 trtllm-serve /opt/vonk/bin/trtllm-serve
COPY llm-api-config.yml /opt/vonk/trtllm/llm-api-config.yml
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
