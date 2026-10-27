FROM nvcr.io/nvidia/vllm@sha256:d049bead397430803ac5b705d50094492f23782716677d746ab721e43f054cf6

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.version="26.08" \
      io.vonk.upstream-image="vllm 0.27.1+93523f72 (NVIDIA NGC vllm 26.08-py3)"

COPY --chmod=0755 vllm /opt/vonk/bin/vllm
COPY harmony-reqs/ /opt/vonk/harmony-reqs/
COPY verify-runtime.py /opt/vonk/verify-runtime.py
RUN echo "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d  /opt/vonk/harmony-reqs/o200k_base.tiktoken" | sha256sum -c - \
    && echo "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7  /opt/vonk/harmony-reqs/cl100k_base.tiktoken" | sha256sum -c - \
    && python3 /opt/vonk/verify-runtime.py

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
