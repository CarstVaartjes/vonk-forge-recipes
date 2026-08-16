FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

LABEL org.opencontainers.image.source="https://github.com/huggingface/diffusers" \
      org.opencontainers.image.revision="a3608b512ed7248499a44c61d954965ed9bdae4d" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.media.adapter="upstream-native"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        diffusers==0.39.0 \
        accelerate==1.10.1 \
        Pillow==11.3.0 \
        safetensors==0.6.2 \
        sentencepiece==0.2.0 \
        transformers==4.57.1

COPY run.py /opt/vonk/source/run.py

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/diffusers-job \
    && cp /opt/vonk/bin/diffusers-job /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/diffusers-job /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
