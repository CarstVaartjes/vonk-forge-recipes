FROM nvcr.io/nvidia/tensorrt-llm/release@sha256:4f30c464ead64fb9727a24064b25057dacc07bef848022421108e544c91f0965

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TensorRT-LLM" \
      org.opencontainers.image.revision="b9ce4b69d12fe5ba65d13893111b1a2ea29413ee" \
      org.opencontainers.image.version="1.3.0rc13" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --chmod=0755 trtllm-serve /opt/vonk/bin/trtllm-serve
COPY llm-api-config.yml /opt/vonk/trtllm/llm-api-config.yml
COPY harmony-reqs/ /opt/vonk/harmony-reqs/
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN echo "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d  /opt/vonk/harmony-reqs/o200k_base.tiktoken" | sha256sum -c - \
    && echo "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7  /opt/vonk/harmony-reqs/cl100k_base.tiktoken" | sha256sum -c - \
    && python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
