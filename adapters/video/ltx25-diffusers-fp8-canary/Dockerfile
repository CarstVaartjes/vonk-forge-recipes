FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

LABEL org.opencontainers.image.source="https://huggingface.co/Lightricks/LTX-2.5-Diffusers" \
      org.opencontainers.image.revision="426936f8b22dc28e4def61e515478b0b7e4a53cc" \
      org.opencontainers.image.licenses="LTX-2 Community License Agreement" \
      io.vonk.diffusers.revision="d035dcd7cc7c88e0a154609b62887d50bba9fdc2" \
      io.vonk.ltx25.profile="fp8-cast-sequential-offload"

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/outputs/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    HOME=/outputs \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_HOME=/outputs/cache/torch \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME=/outputs/cache

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        ffmpeg \
        git \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0 \
        torchvision==0.28.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "diffusers @ git+https://github.com/huggingface/diffusers.git@d035dcd7cc7c88e0a154609b62887d50bba9fdc2" \
        accelerate==1.14.0 \
        av==18.1.0 \
        Pillow==12.3.0 \
        safetensors==0.8.0 \
        sentencepiece==0.2.2 \
        transformers==5.15.1

COPY run.py /opt/vonk/source/run.py
COPY README.md /opt/vonk/source/LTX-2.5-PREFLIGHT.md

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/diffusers-job \
    && chmod 0755 /opt/vonk/bin/diffusers-job \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
