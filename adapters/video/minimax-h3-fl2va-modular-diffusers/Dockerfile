FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://huggingface.co/MiniMaxAI/MiniMax-H3" \
      org.opencontainers.image.revision="42ed227ee7df40d41602854ae760620d6eb651fe" \
      org.opencontainers.image.licenses="MiniMax H3 Community License Agreement" \
      io.vonk.minimax-h3.variant="fl2va-only" \
      io.vonk.diffusers.revision="efabd60d61c2b7aabf9f182bee6b5b6058980304"

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
        "diffusers @ git+https://github.com/huggingface/diffusers.git@efabd60d61c2b7aabf9f182bee6b5b6058980304" \
        accelerate==1.15.0 \
        av==18.1.0 \
        Pillow==12.3.0 \
        safetensors==0.8.0 \
        sentencepiece==0.2.2 \
        torchao==0.18.0 \
        transformers==5.17.0

COPY minimax_h3.py /opt/vonk/source/minimax_h3.py
COPY NOTICE /opt/vonk/source/NOTICE

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/minimax_h3.py "$@"' > /opt/vonk/bin/diffusers-job \
    && chmod 0755 /opt/vonk/bin/diffusers-job \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
