FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/ModelTC/Qwen-Image-Lightning" \
      org.opencontainers.image.revision="25d1e9935ba97e1f60503f8c000201e08b1fe8f4" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.diffusers.revision="c5469b7ceb606edd7ba6570dcd17d38590a18db6"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        git \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.14.0 \
        torchvision==0.29.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "diffusers @ git+https://github.com/huggingface/diffusers.git@c5469b7ceb606edd7ba6570dcd17d38590a18db6" \
        accelerate==1.15.0 \
        Pillow==12.3.0 \
        safetensors==0.8.0 \
        sentencepiece==0.2.2 \
        transformers==5.17.0

COPY qwen_image_lightning.py /opt/vonk/source/qwen_image_lightning.py

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/qwen_image_lightning.py "$@"' > /opt/vonk/bin/diffusers-job \
    && chmod 0755 /opt/vonk/bin/diffusers-job \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
