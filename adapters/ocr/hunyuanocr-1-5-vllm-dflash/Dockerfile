FROM docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c

LABEL org.opencontainers.image.source="https://github.com/Tencent-Hunyuan/HunyuanOCR" \
      org.opencontainers.image.revision="c55965d3da1e6f41987abec8068f2e70851318bc" \
      org.opencontainers.image.licenses="LicenseRef-Tencent-Hunyuan-Community" \
      io.vonk.adapter="hunyuanocr-1-5-vllm-dflash"

USER root

RUN curl --fail --location --retry 5 \
        https://github.com/Tencent-Hunyuan/HunyuanOCR/archive/c55965d3da1e6f41987abec8068f2e70851318bc.tar.gz \
        --output /tmp/hunyuanocr.tar.gz \
    && echo "00789421f6f35aa06635d6a42ded6b0aabd0db7f87d49eded3ce9696018e0d3f  /tmp/hunyuanocr.tar.gz" | sha256sum --check --strict \
    && install --directory --mode=0755 /opt/hunyuanocr /opt/vonk/bin /opt/vonk/source \
    && tar --extract --gzip --file /tmp/hunyuanocr.tar.gz --strip-components=1 --directory /opt/hunyuanocr \
    && rm /tmp/hunyuanocr.tar.gz \
    && chown -R 10001:10001 /opt/vonk /opt/hunyuanocr

COPY run.py /opt/vonk/source/run.py
COPY NOTICE /opt/vonk/source/NOTICE

RUN printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline /opt/vonk/source/run.py \
    && chown -R 10001:10001 /opt/vonk

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/vonk-cache

USER 10001:10001
WORKDIR /opt/hunyuanocr
ENTRYPOINT []
