FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/Tencent-Hunyuan/HunyuanOCR" \
      org.opencontainers.image.revision="1ef4179e53cf860f7e6fd8276a292e5d05b3e927" \
      org.opencontainers.image.licenses="LicenseRef-Tencent-Hunyuan-Community" \
      io.vonk.adapter="hunyuanocr-1-5-vllm-dflash"

USER root

RUN curl --fail --location --retry 5 \
        https://github.com/Tencent-Hunyuan/HunyuanOCR/archive/1ef4179e53cf860f7e6fd8276a292e5d05b3e927.tar.gz \
        --output /tmp/hunyuanocr.tar.gz \
    && echo "b8b5c3514610408e9812903d231393e5b3d5969c9a8fbb9b86caacabc0711b5a  /tmp/hunyuanocr.tar.gz" | sha256sum --check --strict \
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
