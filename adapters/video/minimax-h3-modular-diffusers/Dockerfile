FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

LABEL org.opencontainers.image.source="https://huggingface.co/MiniMaxAI/MiniMax-H3" \
      org.opencontainers.image.revision="42ed227ee7df40d41602854ae760620d6eb651fe" \
      org.opencontainers.image.licenses="MiniMax H3 Community License Agreement" \
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
        torch==2.13.0 \
        torchvision==0.28.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "diffusers @ git+https://github.com/huggingface/diffusers.git@efabd60d61c2b7aabf9f182bee6b5b6058980304" \
        accelerate==1.14.0 \
        av==18.1.0 \
        Pillow==12.3.0 \
        safetensors==0.8.0 \
        sentencepiece==0.2.2 \
        torchao==0.18.0 \
        transformers==5.15.1

COPY minimax_h3.py /opt/vonk/source/minimax_h3.py
COPY NOTICE /opt/vonk/source/NOTICE

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/minimax_h3.py "$@"' > /opt/vonk/bin/diffusers-job \
    && chmod 0755 /opt/vonk/bin/diffusers-job \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
